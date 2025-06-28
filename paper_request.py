import requests
import json
import pandas as pd
from datetime import datetime
import time
import os
from tqdm import tqdm

class OpenAlexSearch:
    """
    A class to search for documents in OpenAlex based on topics, keywords, and date range.
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
        # Combine topics into a search query using OR operator
        query = " OR ".join([f'"{topic}"' for topic in topics])
        
        # Perform the search using the generic search method
        return self._search(query, start_date, end_date, max_results, filters, "topics")
    
    def search_keywords(self, keywords, start_date, end_date, max_results=None, filters=None, keyword_field='default'):
        """
        Search for documents containing specific keywords within a date range.
        
        Args:
            keywords (list): List of keyword strings to search for
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            max_results (int, optional): Maximum number of results to return
            filters (dict, optional): Additional filters to apply
            keyword_field (str, optional): Field to search in ('default', 'title', 'abstract', 'fulltext')
                - 'default': Searches across titles, abstracts, and fulltext
                - 'title': Searches only in titles
                - 'abstract': Searches only in abstracts
                - 'fulltext': Searches only in fulltext
            
        Returns:
            list: List of document dictionaries
        """
        # Field-specific search - map user input to OpenAlex filter fields
        field_map = {
            'default': 'search',           # General search across all fields
            'title': 'filter.title.search', # Title-specific search
            'abstract': 'filter.abstract.search', # Abstract-specific search
            'fulltext': 'filter.fulltext.search'  # Fulltext-specific search
        }
        
        if keyword_field not in field_map:
            raise ValueError(f"Invalid keyword_field: {keyword_field}. Must be one of {list(field_map.keys())}")
        
        # Combine keywords into a search query using OR operator
        query = " OR ".join([f'"{keyword}"' for keyword in keywords])
        
        # Perform the search using the generic search method with field-specific parameters
        return self._search(query, start_date, end_date, max_results, filters, "keywords", keyword_field, field_map[keyword_field])
    
    def _search(self, query, start_date, end_date, max_results=None, filters=None, search_type="topics", keyword_field=None, search_param="search"):
        """
        Generic search method used by both topic and keyword searches.
        
        Args:
            query (str): Formatted search query
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            max_results (int, optional): Maximum number of results to return
            filters (dict, optional): Additional filters to apply
            search_type (str): Type of search (topics or keywords) for logging
            keyword_field (str, optional): Field being searched (for keyword searches)
            search_param (str): API parameter to use for the search query
            
        Returns:
            list: List of document dictionaries
        """
        # Validate input dates
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Dates must be in YYYY-MM-DD format")
            
        # Set up base parameters
        search_params = self.params.copy()
        search_params.update({
            "filter": f"from_publication_date:{start_date},to_publication_date:{end_date}",
            "per-page": 200  # Maximum allowed per page
        })
        
        # Add the search query to the appropriate parameter based on search type
        if search_param == "search":
            search_params[search_param] = query
        else:
            # For field-specific searches like title.search, abstract.search, etc.
            search_params[search_param] = query
        
        # Add any additional filters
        if filters:
            filter_str = search_params["filter"]
            for key, value in filters.items():
                filter_str += f",{key}:{value}"
            search_params["filter"] = filter_str
        
        all_results = []
        page = 1
        total_results = float('inf')  # Initially unknown
        
        search_desc = f"{search_type} search"
        if keyword_field and search_type == "keywords":
            search_desc = f"keywords search in {keyword_field}"
        
        print(f"Searching OpenAlex for: {query}")
        print(f"Search type: {search_desc}")
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
            exclude_words (list, optional): Words to exclude from results (filtering by title)
            format (str): Format to save ('csv', 'excel', or 'json')
        """
        # Convert to DataFrame if needed
        if isinstance(documents, list):
            df = self.extract_key_metadata(documents)
        else:
            df = documents

        # If exclude_words != None remove all documents which have any of the excluded words in their title
        if exclude_words is not None:
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
        "Social amplification of risk", 
        "Nuclear Issues and Defense",
        "Supply Chain Resilience and risk management",
        "Climate Change and geoengineering",
        "Agricultural risk and resilience",
        "Infrastructure resilience and vulnerability analysis",
        "Complex systems and decision making",
        "Global Peace and Security Dynamics",
        "Influence of Climate on Human Conflict",
        "Disaster Management and Resilience",
        "Evolutionary Game Theory and cooperation",
        "World Systems and global transformation",
        "Culture, economy, and development studies",
        "Historical economic and social studies",
        "Military and defense studies",
        "Military history and strategy",
        "Political conflict and governance",
        "Complex Systems and Time Series Analysis",
        "Supply Chain Resilience and Risk Management",
    ]
    
    # Define your keywords of interest
    catastrophic_risk_keywords = [
        "global catastrophic risk",
        "nuclear winter",
        "civilization collapse",
        "societal collapse",
        "historical collapse",
       # "Geopolitics",
       # "systematic risk",


    ]
    # read file email.txt and extract the email address
    with open("email.txt", "r") as file:
        email = file.read().strip()

    # Initialize the searcher with your email for the polite pool
    searcher = OpenAlexSearch(email=email)
    
    start_date = "2025-05-28"
    end_date = "2025-06-28"

    # Search for documents by topic
    topic_results = searcher.search_topics(
        topics=catastrophic_risk_topics,
        start_date=start_date,
        end_date=end_date,
        max_results=300
    )
    
    # Search for documents by keyword (in all fields)
    keyword_results = searcher.search_keywords(
        keywords=catastrophic_risk_keywords,
        start_date=start_date,
        end_date=end_date,
        max_results=200
    )

    # Convert to DataFrame for analysis
    topic_df = searcher.extract_key_metadata(topic_results)
    keyword_df = searcher.extract_key_metadata(keyword_results)

    # Combine the two DataFrames
    combined_df = pd.concat([topic_df, keyword_df], ignore_index=True)
    combined_df.drop_duplicates(subset=['id'], inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
 
    # Save results
    searcher.save_results(combined_df, "collapse_and_GCR_research.csv", exclude_words=[" AI ", "Artificial Intelligence", " AI", "AI "])
