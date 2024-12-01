import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


def visualize_review_analysis(
    reviews: list[str],
    topic_distributions: list[list[tuple[int, float]]],
    dominant_topics: list[int],
    topic_labels: list[str],
    sentiments: list[str],
    topic_names: dict[int, str],
    model_choice: str = "logistic"
):
    """
    Visualizes the review analysis, including topic distributions, sentiment distribution,
    and individual topic distributions for each review, based on the selected sentiment analysis model.

    Args:
        reviews (list[str]): List of reviews.
        topic_distributions (list[list[tuple[int, float]]]): Topic distributions for each review.
        dominant_topics (list[int]): Dominant topic IDs for each review.
        topic_labels (list[str]): Dominant topic names for each review.
        sentiments (list[str]): Sentiments for each review (Positive/Negative/Neutral).
        topic_names (dict[int, str]): Mapping of topic IDs to topic names.
        model_choice (str): Selected sentiment analysis model ('bert' or 'logistic').
    """
    # Display chosen model
    st.write(f"### Sentiment Analysis Using: {model_choice.capitalize()} Model")

    # 1. Display Reviews with Dominant Topics and Sentiments
    st.write("### Review Analysis")
    for i, review in enumerate(reviews):
        st.write(f"**Review {i+1}:** {review}")
        st.write(f"- **Dominant Topic:** {topic_labels[i]} (Topic ID: {dominant_topics[i]})")
        st.write(f"- **Sentiment (via {model_choice.capitalize()}):** {sentiments[i]}")
        st.write("---")

    # 2. Overall Topic Distribution
    st.write("### Overall Topic Distribution")
    topic_distribution_summary = {name: 0 for name in topic_names.values()}
    for dist in topic_distributions:
        for topic_id, prob in dist:
            topic_name = topic_names[topic_id]
            topic_distribution_summary[topic_name] += prob

    # Bar chart for overall topic distribution
    topic_summary_df = pd.DataFrame.from_dict(
        topic_distribution_summary, orient="index", columns=["Probability"]
    ).sort_values(by="Probability", ascending=False)
    st.bar_chart(topic_summary_df, use_container_width=True)

    # 3. Sentiment Distribution
    st.write("### Sentiment Distribution")
    # Ensure sentiment summary includes all categories even if counts are zero
    sentiment_summary = {
        "Positive": sentiments.count("Positive"),
        "Negative": sentiments.count("Negative"),
        "Neutral": sentiments.count("Neutral"),
    }
    sentiment_df = pd.DataFrame.from_dict(
        sentiment_summary, orient="index", columns=["Count"]
    ).sort_values(by="Count", ascending=False)
    st.bar_chart(sentiment_df, use_container_width=True)

    # 4. Individual Topic Distributions for Each Review
    st.write("### Topic Distribution per Review")
    for i, (review, dist) in enumerate(zip(reviews, topic_distributions)):
        st.write(f"#### Review {i+1}: {review}")

        # Convert sparse topic distribution to named format
        named_dist = {topic_names[topic_id]: prob for topic_id, prob in dist}
        dist_df = pd.DataFrame.from_dict(
            named_dist, orient="index", columns=["Probability"]
        ).sort_values(by="Probability", ascending=False)

        # Display bar chart
        st.bar_chart(dist_df, use_container_width=True)

    # 5. Optional: Add a Combined Pie Chart for Sentiment Distribution
    st.write("### Overall Sentiment Distribution (Pie Chart)")
    fig, ax = plt.subplots()
    ax.pie(
        sentiment_summary.values(),
        labels=sentiment_summary.keys(),
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.axis("equal")  # Equal aspect ratio ensures the pie chart is circular.
    st.pyplot(fig)
