import requests
import json
import pandas as pd
from datetime import datetime
import time
import argparse
import os
from tqdm import tqdm

class OpenAlexSearch:
    """
    A class to search for documents in OpenAlex based on topics and date range.
    Uses the OpenAlex API to perform searches and retrieve document metadata.
    """
    
    def __init__(self, email=None, api_key=None):
        """
        Initialize the OpenAlex search with optional authentication.
        
        Args:
            email (str, optional): Email for the polite pool. 
            api_key (str, optional): API key for premium access.
        """
        self.base_url = "https://api.openalex.org"
        self.headers = {}
        
        # Set up authentication parameters
        self.params = {}
        if email:
            self.params["mailto"] = email
        if api_key:
            self.params["api_key"] = api_key
        
        # Set up session for connection pooling
        self.session = requests.Session()
    
    def search_topics(self, topics, start_date, end_date, max_results=None, filters=None):
        """
        Search for documents on specified topics within a date range.
        
        Args:
            topics (list): List of topic strings to search for
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            max_results (int, optional): Maximum number of results to return
            filters (dict, optional): Additional filters to apply
            
        Returns:
            list: List of document dictionaries
        """
        # Validate input dates
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Dates must be in YYYY-MM-DD format")
            
        # Combine topics into a search query using OR operator
        query = " OR ".join([f'"{topic}"' for topic in topics])
        
        # Set up base parameters
        search_params = self.params.copy()
        search_params.update({
            "filter": f"from_publication_date:{start_date},to_publication_date:{end_date}",
            "search": query,
            "per-page": 200  # Maximum allowed per page
        })
        
        # Add any additional filters
        if filters:
            filter_str = search_params["filter"]
            for key, value in filters.items():
                filter_str += f",{key}:{value}"
            search_params["filter"] = filter_str
        
        all_results = []
        page = 1
        total_results = float('inf')  # Initially unknown
        
        print(f"Searching OpenAlex for: {query}")
        print(f"Date range: {start_date} to {end_date}")
        
        with tqdm(desc="Retrieving results", unit="docs") as pbar:
            while (len(all_results) < total_results) and (not max_results or len(all_results) < max_results):
                # Update page parameter
                search_params["page"] = page
                
                # Make request with exponential backoff
                response = self._make_request_with_backoff(
                    f"{self.base_url}/works", 
                    params=search_params
                )
                
                if response.status_code != 200:
                    print(f"Error: {response.status_code}")
                    print(response.text)
                    break
                
                data = response.json()
                
                # Update our understanding of total available results
                total_results = data["meta"]["count"]
                pbar.total = min(total_results, max_results or float('inf'))
                
                # Get results from this page
                results = data["results"]
                all_results.extend(results)
                pbar.update(len(results))
                
                # If we've reached max_results, truncate and break
                if max_results and len(all_results) >= max_results:
                    all_results = all_results[:max_results]
                    break
                
                # If no more results, break
                if len(results) == 0 or "next_cursor" not in data["meta"] or not data["meta"]["next_cursor"]:
                    break
                
                # Move to next page
                page += 1
                
                # Respect rate limits
                time.sleep(0.1)
        
        print(f"Retrieved {len(all_results)} of {total_results} total results")
        return all_results
    
    def _make_request_with_backoff(self, url, params=None, max_retries=5):
        """
        Makes a request with exponential backoff to handle rate limiting.
        
        Args:
            url (str): URL to request
            params (dict, optional): Query parameters
            max_retries (int, optional): Maximum number of retries
            
        Returns:
            Response: The response object
        """
        retry = 0
        while retry < max_retries:
            response = self.session.get(url, params=params, headers=self.headers)
            
            # If successful or not a rate limit error, return
            if response.status_code != 429:
                return response
            
            # Rate limited, wait with exponential backoff
            wait_time = 2 ** retry
            print(f"Rate limited. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            retry += 1
        
        # If we've exhausted retries, return the last response
        return response
    
    def extract_key_metadata(self, documents):
        """
        Extract key metadata from documents for easier analysis.
        
        Args:
            documents (list): List of document dictionaries from OpenAlex
            
        Returns:
            pd.DataFrame: DataFrame with key metadata
        """
        metadata = []
        
        for doc in documents:
            # Extract basic metadata
            item = {
                'id': doc.get('id', ''),
                'title': doc.get('title', ''),
                'publication_date': doc.get('publication_date', ''),
                'doi': doc.get('doi', ''),
                'type': doc.get('type', ''),
                'cited_by_count': doc.get('cited_by_count', 0),
            }
            
            # Extract journal/source
            if 'primary_location' in doc and doc['primary_location'] and 'source' in doc['primary_location']:
                source = doc['primary_location']['source']
                item['source_name'] = source.get('display_name', '') if source else ''
            else:
                item['source_name'] = ''
            
            # Extract authors
            if 'authorships' in doc and doc['authorships']:
                authors = [a.get('author', {}).get('display_name', '') for a in doc['authorships'] if 'author' in a]
                item['authors'] = '; '.join(authors)
                
                # Extract affiliations from first author
                if doc['authorships'][0].get('institutions'):
                    affiliations = [i.get('display_name', '') for i in doc['authorships'][0].get('institutions', [])]
                    item['first_author_affiliations'] = '; '.join(affiliations)
                else:
                    item['first_author_affiliations'] = ''
            else:
                item['authors'] = ''
                item['first_author_affiliations'] = ''
            
            # Extract open access info
            if 'open_access' in doc:
                item['is_oa'] = doc['open_access'].get('is_oa', False)
                item['oa_status'] = doc['open_access'].get('oa_status', '')
                item['oa_url'] = doc['open_access'].get('oa_url', '')
            else:
                item['is_oa'] = False
                item['oa_status'] = ''
                item['oa_url'] = ''
            
            # Extract abstract if available
            if 'abstract_inverted_index' in doc and doc['abstract_inverted_index']:
                # Convert inverted index back to text (simplified approach)
                try:
                    index = doc['abstract_inverted_index']
                    max_pos = max([pos for positions in index.values() for pos in positions])
                    words = [''] * (max_pos + 1)
                    
                    for word, positions in index.items():
                        for pos in positions:
                            words[pos] = word
                    
                    item['abstract'] = ' '.join(words)
                except:
                    item['abstract'] = '[Abstract conversion failed]'
            else:
                item['abstract'] = ''
            
            # Extract topics
            if 'topics' in doc and doc['topics']:
                topics = [t.get('display_name', '') for t in doc['topics']]
                item['topics'] = '; '.join(topics)
            else:
                item['topics'] = ''
            
            metadata.append(item)
        
        # Convert to DataFrame
        df = pd.DataFrame(metadata)
        return df
    
    def save_results(self, documents, filepath, exclude_words=None, format='csv'):
        """
        Save search results to a file.
        
        Args:
            documents (list or DataFrame): Documents to save
            filepath (str): Path to save file
            format (str): Format to save ('csv', 'excel', or 'json')
        """
        # Convert to DataFrame if needed
        if isinstance(documents, list):
            df = self.extract_key_metadata(documents)
        else:
            df = documents

        # if exclude_words != none remove all documents which have any of the excluded words in their title
        if exclude_words != None:
            df = filter_dataframe(df, exclude_words)                
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save in requested format
        if format.lower() == 'csv':
            df.to_csv(filepath, index=False, encoding='utf-8')
        elif format.lower() == 'excel':
            df.to_excel(filepath, index=False)
        elif format.lower() == 'json':
            if isinstance(documents, list):
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(documents, f, ensure_ascii=False, indent=2)
            else:
                df.to_json(filepath, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Results saved to {filepath}")


def filter_dataframe(df, words_to_exclude):
    """
    Remove rows where title contains any word from a list of words.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with a 'title' column to filter
    words_to_exclude : list
        List of words to check for in titles
    
    Returns:
    -----------
    pandas.DataFrame
        Filtered DataFrame with matching rows removed
    """
    # Create a copy to avoid modifying the original DataFrame
    filtered_df = df.copy()
    
    # Convert all titles and words to lowercase for case-insensitive matching
    lowercase_titles = filtered_df['title'].str.lower()
    lowercase_words = [word.lower() for word in words_to_exclude]
    
    # Create a boolean mask for rows to keep
    # A row is kept if its title doesn't contain any of the words to exclude
    mask = ~lowercase_titles.apply(lambda title: any(word in title for word in lowercase_words))
    
    # Return the filtered DataFrame
    return filtered_df[mask]


# Example usage
if __name__ == "__main__":
    # Example direct usage (uncomment to use)
    # Define your topics of interest
    catastrophic_risk_topics = [
        "Global Catastrophic Risk", 
        "Societal Collapse", 
        "Nuclear War", 
        "Supervolcanic eruption",
        "Resilience global catastrophe", 
        "Social amplification of risk", 
        #"Existential risk"
    ]
    
    # Initialize the searcher with your email for the polite pool
    searcher = OpenAlexSearch(email="xtzbo96mr@mozmail.com")
    
    # Search for documents
    results = searcher.search_topics(
        topics=catastrophic_risk_topics,
        start_date="2025-01-01",
        end_date="2025-03-20",
        max_results=500
    )
    
    # Convert to DataFrame for analysis
    results_df = searcher.extract_key_metadata(results)
    
    # Save results
    searcher.save_results(results_df, "catastrophic_risk_research.csv", exclude_words=[" AI ", "Artificial Intelligence", " AI", "AI "])