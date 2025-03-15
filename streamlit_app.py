import streamlit as st
import gensim.downloader
import spacy
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import plotly.express as px
import pandas as pd
import numpy as np
import nltk
import plotly.graph_objs as go
from sklearn import datasets

nltk.download('stopwords')

# Text Model
@st.cache_resource
def loading_model():
    model = gensim.downloader.load('glove-twitter-25')  # Replace with your desired model
    return model

def paragraph_tokenization(paragraph):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(paragraph)

    # Tokenize and clean the text
    tokens = [
        token.text.lower() for token in doc
        if token.text.lower() not in stopwords.words('english')  # Remove stopwords
        and not token.is_punct  # Remove punctuation
        and not token.is_space  # Remove spaces
    ]
    return tokens

def tokens2vectors(tokens, model):
    word2vector_dict = {}
    for token in tokens:
        try:
            word2vector_dict[token] = model[token]
        except KeyError:
            pass
    
    vectors = np.array(list(word2vector_dict.values()))
    return vectors, list(word2vector_dict.keys())

def reduce_dimensions(vectors, method="tsne", iterations=None):
    if method == "tsne":
        perplexity = min(30, len(vectors) - 1)
        if iterations is None:
            iterations = 1
        tsne = TSNE(n_components=3, perplexity=perplexity, max_iter=max(iterations, 250), random_state=42, init="pca")
        reduced_vectors = tsne.fit_transform(vectors)
    elif method == "pca":
        pca = PCA(n_components=3, random_state=42)
        reduced_vectors = pca.fit_transform(vectors)
    elif method == "umap":
        reducer = umap.UMAP(n_components=3, n_jobs=-1, n_epochs=500)
        reduced_vectors = reducer.fit_transform(vectors)
    else:
        raise ValueError("Unknown dimensionality reduction method.")
    return reduced_vectors

def scatter_plot(reduced_vectors, tokens):
    # Convert reduced_vectors to a DataFrame
    df = pd.DataFrame(reduced_vectors, columns=['x', 'y', 'z'])
    df["tokens"] = tokens
    df["tokens"] = df["tokens"].astype(str)
    # Create the scatter plot
    fig = px.scatter_3d(df, x='x', y='y', z='z',
        text='tokens', opacity=0.8,
        color_discrete_sequence=px.colors.qualitative.G10  # Optional: display token names
    )
    # Adjust the figure size
    fig.update_layout(
        autosize=True,
        height=900,
        scene_aspectmode='cube',  # Adjust height
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        )
    )
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)

# Image Model
def reduce_dimensions_with_animation(vectors, method="tsne", iterations=500):
    if method != "tsne":
        st.sidebar.write("Animation is only available for t-SNE.")
        return None
    
    st.sidebar.write("Performing dimensionality reduction with t-SNE...")

    # Initialize t-SNE with partial fitting to capture intermediate steps
    tsne = TSNE(
        n_components=3,
        perplexity=min(30, len(vectors) - 1),
        max_iter=iterations,
        init="pca",
        random_state=42,
        method="barnes_hut",
        learning_rate="auto",
        verbose=1
    )
    
    # Fit t-SNE and store intermediate results
    intermediate_results = []
    for i in range(250, iterations, 50):  # Capture results every 50 iterations
        tsne.max_iter = i
        reduced_vectors = tsne.fit_transform(vectors)
        intermediate_results.append(reduced_vectors)
    
    st.sidebar.write("t-SNE dimensionality reduction completed.")
    print(f'This is ********************************************{intermediate_results}')
    return intermediate_results

def image_scatter_plot(reduced_vectors, tokens):
    # Convert reduced_vectors to a DataFrame
    df = pd.DataFrame(reduced_vectors, columns=['x', 'y', 'z'])
    df["target"] = tokens
    df["target"] = df["target"].astype(str)
    # Create the scatter plot
    fig = px.scatter_3d(df, x='x', y='y', z='z',
        text='target', color='target', opacity=0.8,
        color_discrete_sequence=px.colors.qualitative.G10  # Optional: display token names
    )
    # Adjust the figure size
    fig.update_layout(
        autosize=True,
        height=900,
        scene_aspectmode='cube',  # Adjust height
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        )
    )
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)

def plot_animation(intermediate_results, tokens):
    st.sidebar.write("Creating t-SNE animation...")
    
    # Prepare frames for animation
    frames = []
    for i, reduced_vectors in enumerate(intermediate_results):
        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=reduced_vectors[:, 0],
                    y=reduced_vectors[:, 1],
                    z=reduced_vectors[:, 2],
                    mode="markers",
                    marker=dict(size=5, color=tokens),
                    text=tokens,  # Annotate with tokens
                )
            ],
            name=f"Step {i*50}"  # Frame name
        )
        frames.append(frame)

    # Define the layout of the animation
    layout = go.Layout(
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis",
        ),
        title="t-SNE Dimensionality Reduction Animation",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 200, "redraw": True}}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": True}}]),
                ],
            )
        ],
    )

    # Combine frames into a figure
    fig = go.Figure(
        data=frames[0].data,  # Initialize with the first frame
        frames=frames,  # Add all frames
        layout=layout,
    )
    
    st.sidebar.write("Animation created successfully.")
    st.plotly_chart(fig, use_container_width=True)


# Main Streamlit app
def main():
    # Set page title and icon
    st.set_page_config(
        page_title="3D Embedding Visualizer",
        page_icon="thumbnail/3d_embedding_viz.png",
        layout="wide"
    )

    margins_css = """
        <style>
            [data-testid="stSidebarHeader"] {
                padding-bottom: 0rem;
                padding-right: 0rem;
            }
            .st-emotion-cache-hzo1qh {
                left: 0rem;
            }
            [data-testid="stMainBlockContainer"] {
                padding-bottom:0rem;            
            }
        </style>
    """
    st.markdown(margins_css, unsafe_allow_html=True)

    # Sidebar Title
    st.sidebar.title("3D Embedding Visualizer")

    # Sidebar for model and dimensionality reduction method
    selected_type = st.sidebar.segmented_control("Data Type:", ['Image', 'Text'], )
    dr_method = st.sidebar.selectbox("Dimensionality Reduction Method", ["pca", "tsne", "umap"])
    iterations = st.sidebar.slider("Iterations (for TSNE)", min_value=250, max_value=3000, step=10, value=500)

    if selected_type=='Image':
        # User input
        st.sidebar.title("Select Dataset:")
        selected_type = st.sidebar.segmented_control("Image Dataset:", ['IRIS', 'MNIST', 'BREAST CANCER', 'OLIVERTTI FACES'], label_visibility='collapsed')

        if st.sidebar.button("Visualize Embeddings", type='primary'):

            if selected_type=="IRIS":

                X, y = datasets.load_iris(return_X_y=True)

                reduced_vectors = reduce_dimensions(X, method=dr_method, iterations=iterations if dr_method == "tsne" else None)
                # Plotting
                # status.write('Creating 3D scatter plot...')
                image_scatter_plot(reduced_vectors, y)

            elif selected_type=="MNIST":

                X, y = datasets.load_digits(return_X_y=True)

                reduced_vectors = reduce_dimensions(X, method=dr_method, iterations=iterations if dr_method == "tsne" else None)
                # Plotting
                # status.write('Creating 3D scatter plot...')
                image_scatter_plot(reduced_vectors, y)

            elif selected_type=="BREAST CANCER":

                X_digits, y_digits = datasets.load_breast_cancer(return_X_y=True)

                reduced_vectors = reduce_dimensions(X_digits, method=dr_method, iterations=iterations if dr_method == "tsne" else None)
                # Plotting
                # status.write('Creating 3D scatter plot...')
                image_scatter_plot(reduced_vectors, y_digits)

            elif selected_type=="OLIVERTTI FACES":

                X_digits, y_digits = datasets.fetch_olivetti_faces(return_X_y=True)

                reduced_vectors = reduce_dimensions(X_digits, method=dr_method, iterations=iterations if dr_method == "tsne" else None)
                # Plotting
                # status.write('Creating 3D scatter plot...')
                image_scatter_plot(reduced_vectors, y_digits)

            else:
                st.warning("Select a Dataset")


        #         # Applying Dimensionality Reduction
        #         if len(words)>7:
        #             status.write(f'Applying Dimensionality Reduction...')
        #             reduced_vectors = reduce_dimensions(vectors, method=dr_method, iterations=iterations if dr_method == "tsne" else None)
        #             # Plotting
        #             status.write('Creating 3D scatter plot...')
        #             scatter_plot(reduced_vectors, words)
        #             status.update(label="Created 3D Embedding Space!", state="complete", expanded=False)
        #         else:
        #             status.update(label="Need a longer text!", state='error', expanded=True)
        #             st.warning('Enter a longer paragraph')
        #     else:
        #         st.warning("Please enter a paragraph.")






        
        # # Example vector and tokens (replace with real data)
        # vectors = np.random.rand(100, 50)  # Example high-dimensional data
        # tokens = [f"Token {i}" for i in range(100)]

        # intermediate_results = reduce_dimensions_with_animation(vectors, method="tsne", n_iter=500)
        # if intermediate_results:
        #     plot_animation(intermediate_results, tokens)

    elif selected_type=='Text':
        # User input
        st.sidebar.title("Input Paragraph:")
        paragraph = st.sidebar.text_area("Enter a paragraph:", placeholder="Enter your text here...", label_visibility='collapsed')

        if st.sidebar.button("Visualize Embeddings", type='primary'):
            if paragraph.strip():

                status = st.sidebar.status("Generating 3D Word Embedding...", expanded=True)

                # Loading Model
                status.write('Loading Gensim model...')
                model = loading_model()
                # Tokenizing Paragraph
                status.write('Tokenizing...')
                tokens = paragraph_tokenization(paragraph)
                # Word embedding                    
                status.write('Generating word embeddings...')
                vectors, words = tokens2vectors(tokens, model)
                # Applying Dimensionality Reduction
                if len(words)>7:
                    status.write(f'Applying Dimensionality Reduction...')
                    reduced_vectors = reduce_dimensions(vectors, method=dr_method, iterations=iterations if dr_method == "tsne" else None)
                    # Plotting
                    status.write('Creating 3D scatter plot...')
                    scatter_plot(reduced_vectors, words)
                    status.update(label="Created 3D Embedding Space!", state="complete", expanded=False)
                else:
                    status.update(label="Need a longer text!", state='error', expanded=True)
                    st.warning('Enter a longer paragraph')
            else:
                st.warning("Please enter a paragraph.")

    else:
        st.warning('Select Data Type')

if __name__ == "__main__":
    main()