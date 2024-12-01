import streamlit as st
from scraper.scrapy_runner import process_reviews
from visualise import visualize_review_analysis

# Streamlit App Title
st.title("Customer Feedback Analysis")
st.write("Analyze customer reviews for dominant topics and sentiments.")

# Sidebar Inputs
url_input = st.sidebar.text_input("Enter Amazon Product URL", placeholder="https://www.amazon.in/example-product")
max_pages = st.sidebar.slider("Number of Pages to Scrape", 1, 10, 4)
model_choice = st.sidebar.selectbox("Choose Sentiment Analysis Model", ["logistic", "bert"])

# Scrape and Analyze Button
if st.sidebar.button("Scrape and Analyze"):
    if not url_input.strip():
        st.error("Please enter a valid URL.")
    else:
        st.write("### Scraping Reviews...")
        try:
            # Process the reviews with the selected model
            results = process_reviews(url_input, max_pages, model_choice=model_choice, visualize=False)

            # Extract results
            reviews = results.get("reviews", [])
            sentiments = results.get("sentiments", [])
            topic_distributions = results.get("topic_distributions", [])
            dominant_topics = results.get("dominant_topics", [])
            topic_labels = results.get("topic_labels", [])

            # Check if reviews were scraped
            if not reviews:
                st.warning("No reviews were scraped. Please try a different URL or adjust settings.")
            else:
                # Define topic names (optional, can be extended)
                topic_names = {
                    0: "Product Performance and Issues",
                    1: "Ordering and Delivery Experience",
                    2: "Product Quality and Satisfaction",
                }

                # Visualize analysis
                visualize_review_analysis(
                    reviews,
                    topic_distributions,
                    dominant_topics,
                    topic_labels,
                    sentiments,
                    topic_names,
                    model_choice=model_choice,
                )

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
