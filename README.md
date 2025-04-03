# OpenAlex-Paper-Request
This is some code that gets you all documents from OpenAlex via their API for specific topics and keywords and saves them in a csv. 

## How to use this
### Create email 
If you want to have quick access to the OpenAlex API you have to provide an email. This code here looks for an "email.txt" which only contains the email address you want to use. You can also just change the code. 

### Find the keywords/topics you find interesting
OpenAlex has automated topics classification. Unfortunately, they do not provide a complete list of their topics and keywords. In the current version of the code I have added all those I found relevant for global catastrophic risk and societal collapse. They way I found those was to look for authors, papers and topics I thought relevant on the [OpenAlex website](https://openalex.org/). This allows you to see under which topics and keywords the results of your query are categorized. Just pick the ones that seem relevant to you and plug them in the code here. 
