import scrapy
from scrapy.http import Response


class AmazonReviewSpider(scrapy.Spider):
    """
    Spider for scraping reviews from Amazon product pages.

    Attributes:
        name (str): Name of the spider.
        allowed_domains (list[str]): List of allowed domains for scraping.
        start_urls (list[str]): Starting URL(s) for the spider.
        max_pages (int): Maximum number of pages to scrape.
    """

    name: str = "amazon_reviews"
    allowed_domains: list[str] = ["amazon.in"]

    def __init__(self, url: str, max_pages: int, *args, **kwargs):
        """
        Initializes the spider with the given URL and maximum number of pages.

        Args:
            url (str): The starting URL for scraping.
            max_pages (int): Maximum number of pages to scrape.
            *args: Additional arguments for the parent class.
            **kwargs: Additional keyword arguments for the parent class.
        """
        super().__init__(*args, **kwargs)
        self.start_urls = [url]
        self.max_pages = max_pages
        # self.page_count = 0

    def parse(self, response: Response):
        """
        Parses the response to extract review data.

        Args:
            response (Response): Scrapy HTTP response object.

        Yields:
            dict: A dictionary containing extracted review data.
        """
        for review in response.css('div[data-hook="review"]'):
            yield {
                # "reviewer_name": review.css("span.a-profile-name::text").get(),
                # "review_title": review.css('a[data-hook="review-title"] span::text').get(),
                "review_rating": review.css('i[data-hook="review-star-rating"] span::text').get(),
                "review_date": review.css('span[data-hook="review-date"]::text').get(),
                "review_body": review.css('span[data-hook="review-body"] span::text').get(),
                # "verified_purchase": review.css('span[data-hook="avp-badge-linkless"]::text').get(),
            }
        
        # Handle Pagination
        # next_page = response.css(".a-last a::attr(href)").get()
        # self.page_count += 1

        # if next_page and self.page_count < self.max_pages:
        #     self.logger.info(f"Fetching page {self.page_count + 1}")

        #     yield response.follow(next_page, self.parse)

        # else:
        #     self.logger.info(f"Scraped {self.page_count} page(s). Reached the last page or max pages limit.")
