# This Python program is a web application built using the Dash framework. It's designed to find authors within the UNIC alliance based on the abstracts of approximately 80,000 papers published by the UNIC alliance and indexed in Web of Science from 2020 until the end of 2023.

# The application has a sidebar and a main content area. The sidebar contains a download button, some information about the tool, and the author's credit. The main content area contains a Dash DataTable that displays the search results.

# The tool uses the TF-IDF vectorizer to find the most similar abstracts to the search query. The TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer is a technique used in information retrieval to reflect how important a word is to a document in a collection or corpus. The TF-IDF value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently in general.

# The DataTable is set to not have any page actions, and its headers are fixed. The table's style is set to allow vertical and horizontal scrolling, and its height is set to 95% of the container's height. The table's rows have a conditional style that sets the background color to light gray for odd rows. The 'Abstract', 'Article Title', and 'Source Title' columns have a fixed width.

# The download button triggers a download of the data displayed in the DataTable. The dcc.Download component is used to handle the download, and the dcc.Store component is used to store the data that will be downloaded.


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from dash import dash_table
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from dash.exceptions import PreventUpdate

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stopwords and lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)



# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Load the data frame
df = pd.read_pickle('/data/final_df_preprocessed.pkl')



# Fit the vectorizer on the "Abstract" column
tfidf_matrix = vectorizer.fit_transform(df['Abstract_Preprocessed'])

# Create a Dash application
app = dash.Dash(__name__)

# Add external stylesheets
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



# Define the layout of the application
app.layout = html.Div(style={'display': 'flex', 'flex-direction': 'column', 'height': '100vh'}, children=[
    html.Div([
        html.Img(src='/assets/unizg_logo.png'),
        html.Img(src='/assets/unic_logo.png')], style={'display': 'flex', 'justify-content': 'flex-end', 'height': '5%'}), 
    html.Div(style={'display': 'flex', 'height': '95%'}, children=[
        html.Div([
            html.H1("Abstract Search"),
            dcc.Input(id='search-input', type='text', placeholder='Enter the search phrase or words'),
            html.Br(),
            html.Br(),
            html.P("Enter the number of results you want:"),
            dcc.Input(id='num-results-input', type='number', placeholder='Enter the number of results you want', value=30),
            html.Button('Search', id='search-button', style={'backgroundColor': 'lightgray'}),
            html.Br(),
            html.Br(),
            html.Button("Download data", id="btn_xlsx", style={'backgroundColor': 'lightgray'}),
            dcc.Download(id="download-data"),
            dcc.Store(id='store-data'), 
            html.Br(),
            html.Br(),
            html.H2("About"),
            
            html.P("This is a tool for finding authors within the UNIC alliance. The tool is based on the abstracts of the approximately 80.000 papers published by the UNIC alliance indexed in Web of Science from 2020 until the end of 2023. The tool uses the TF-IDF vectorizer to find the most similar abstracts to the search query."),
            html.Br(),
            html.P([html.B("Please be aware of the limitations of the tool, such as WoS errors, affiliation errors, etc. This tool uses TD-IDF cosine simmilarity for searching which is the balance between accuracy and speed. To find more results, increase the number of results")]),
            html.Br(),
            html.P("The tool is developed by the University of Zagreb, Research Office. Authored by Miroslav Rajter.")
        ], style={'display': 'flex', 'flex-direction': 'column', 'justify-content': 'space-around', 'height': '100%', 'width': '20%', 'borderRight': '1px solid black'}),
        html.Div([
                dash_table.DataTable(
                    id='results-table',
                    page_action='none',
                    fixed_rows={'headers': True},
                    style_table={'overflowY': 'auto','overflowX': 'auto', 'height': '95%'},
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        },
                        {
                        'if': {'column_id': 'Abstract'},
                        'width': '300px'
                        },
                        {
                            'if': {'column_id': 'Article Title'},
                            'width': '200px'
                        },
                        {
                            'if': {'column_id': 'Source Title'},
                            'width': '100px'
                        },
                        {
                            'if': {'column_id': 'Publication Year'},
                            'width': '100px'
                        },  
                        {
                            'if': {'column_id': 'Document Type'},
                            'width': '100px'
                        },  
                        {
                            'if': {'column_id': 'Zagreb'},
                            'width': '100px'
                        },
                        {
                            'if': {'column_id': 'Liege'},
                            'width': '100px'
                        },   
                        {
                            'if': {'column_id': 'EUR'},
                            'width': '100px'
                        },    
                        {
                            'if': {'column_id': 'Oulu'},
                            'width': '100px'
                        },  
                        {
                            'if': {'column_id': 'Lodz'},
                            'width': '100px'
                        },     
                        {
                            'if': {'column_id': 'Deusto'},
                            'width': '100px'
                        },
                        {
                            'if': {'column_id': 'Cork'},
                            'width': '100px'
                        },
                        {
                            'if': {'column_id': 'Bochum'},
                            'width': '100px'
                        },
                        {
                            'if': {'column_id': 'Malmo'},
                            'width': '100px'
                        },
                        {
                            'if': {'column_id': 'Koc'},
                            'width': '100px'
                        },
                    ],
                    style_cell={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        'verticalAlign': 'top',
                        'textAlign': 'left'

                    },
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold',
                        'verticalAlign': 'middle',
                        'textAlign': 'center'
                    },
                    style_cell_conditional=[
                        {
                            'if': {'column_id': c},
                            'textAlign': 'left'
                        } for c in ['Date', 'Region']
                    ]

                    )
            ], style={'width': '80%'})
        ])
    ])


@app.callback(
    [Output('results-table', 'data'),
     Output('results-table', 'columns'),
     Output('store-data', 'data')],  # Add an Output for the dcc.Store component
    [Input('search-button', 'n_clicks')],
    [State('search-input', 'value'),
     State('num-results-input', 'value')]
)

def update_results(n_clicks, search_query, num_results):
    if n_clicks is None:
        return dash.no_update
    else:
        # Preprocess the search query
        search_query = preprocess_text(search_query)

        # Transform the search query using the vectorizer
        search_vector = vectorizer.transform([search_query])

        # Calculate the cosine similarity between the search vector and all abstract vectors
        cosine_similarities = cosine_similarity(search_vector, tfidf_matrix).flatten()

        # Get the indices of the abstracts sorted by their similarity
        sorted_indices = cosine_similarities.argsort()

        # Get the indices of the most similar abstracts
        most_similar_indices = sorted_indices[-num_results:]

        # Get the most similar abstracts
        most_similar_abstracts = df.loc[most_similar_indices]

        # Drop the 'Abstract_preprocessed' column
        most_similar_abstracts = most_similar_abstracts.drop(columns=['Abstract_Preprocessed', 'Author Keywords', 'Addresses'])

        # Return the most similar abstracts as a DataFrame
        data = most_similar_abstracts.to_dict('records')
        columns = [{"name": i, "id": i} for i in most_similar_abstracts.columns]
        return data, columns, data


@app.callback(
    Output("download-data", "data"),
    Input("btn_xlsx", "n_clicks"),
    State("store-data", "data"),  # Add a State for the dcc.Store component
    prevent_initial_call=True,
)
def func(n_clicks, data):
    if not n_clicks or not data:
        raise PreventUpdate

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # Return the DataFrame as an Excel file
    return dcc.send_data_frame(df.to_excel, "mydata.xlsx")



# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)