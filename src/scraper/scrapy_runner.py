import os
import json
import multiprocessing
from scrapy.crawler import CrawlerProcess
from .scraper.spiders.amazon_scraping import AmazonReviewSpider
from models.sentiment_analysis import analyse_sentiments
from models.topic_modelling import topic_modelling


def run_spider(url: str, max_pages: int, output_path: str) -> None:
    """
    Runs the Scrapy spider to scrape Amazon reviews.

    Args:
        url (str): The URL of the Amazon product to scrape reviews from.
        max_pages (int): Maximum number of pages to scrape.
        output_path (str): Path to save the scraped reviews as a JSON file.
    """
    process = CrawlerProcess(
        settings={
            "FEEDS": {output_path: {"format": "json"}},
        }
    )
    process.crawl(AmazonReviewSpider, url=url, max_pages=max_pages)
    process.start()


def scrape_amazon_reviews(url: str, max_pages: int) -> list[str]:
    """
    Scrapes Amazon reviews using Scrapy and saves the output in a JSON file.

    Args:
        url (str): The URL of the Amazon product to scrape reviews from.
        max_pages (int): Maximum number of pages to scrape.

    Returns:
        list[str]: A list of review bodies extracted from the scraped data.
    """
    output_path = "data/reviews_json/reviews.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Run the spider in a separate process to avoid ReactorNotRestartable error
    spider_process = multiprocessing.Process(
        target=run_spider, args=(url, max_pages, output_path)
    )
    spider_process.start()
    spider_process.join()

    # Read and return scraped reviews
    with open(output_path, "r") as f:
        reviews = json.load(f)
        
    return [review["review_body"] for review in reviews]


def process_reviews(url: str, max_pages: int = 4, model_choice: str = "logistic", visualize: bool = True) -> dict:
    """
    Scrapes reviews, performs sentiment analysis, and extracts topics.

    Args:
        url (str): The URL of the Amazon product to scrape reviews from.
        max_pages (int, optional): Maximum number of pages to scrape. Defaults to 4.
        model_choice (str, optional): Sentiment analysis model to use ('logistic' or 'bert'). Defaults to 'logistic'.
        visualize (bool, optional): Whether to visualize the topic modeling results. Defaults to True.

    Returns:
        dict: A dictionary containing reviews, sentiments, topic distributions, dominant topics, 
        and topic labels.
    """
    try:
        # Scrape reviews
        reviews = scrape_amazon_reviews(url, max_pages)

        if not reviews:
            raise ValueError("No reviews were scraped. Please check the URL or try a different product.")

        # Analyze sentiments using the chosen model
        sentiments = [analyse_sentiments(review, model_type=model_choice) for review in reviews]

        # Extract topics
        topic_distributions, dominant_topics, topic_labels = topic_modelling(reviews, visualize=visualize)

        # Compile results
        results = {
            "reviews": reviews,
            "sentiments": sentiments,
            "topic_distributions": topic_distributions,
            "dominant_topics": dominant_topics,
            "topic_labels": topic_labels,
        }

        return results

    except Exception as e:
        raise RuntimeError(f"An error occurred while processing reviews: {str(e)}")
