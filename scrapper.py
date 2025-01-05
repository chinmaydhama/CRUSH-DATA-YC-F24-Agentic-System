from playwright.sync_api import sync_playwright
import json

# Function to scrape Notion documentation and structure the content for RAG pipeline
def scrape_notion_page(url):
    with sync_playwright() as p:
        # Launch the browser
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the Notion page
        page.goto(url)

        # Wait for the page to load completely
        page.wait_for_load_state("networkidle")

        # Scroll to the bottom of the page to ensure all content is loaded
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

        # Expand all toggle sections if present
        toggles = page.locator("[data-block-id][role='button']")
        for toggle in toggles.element_handles():
            toggle.click()

        # Extract title
        title = page.title()

        # Initialize structured data storage
        structured_data = {"title": title, "sections": []}

        # Extract main content sections
        content_blocks = page.locator("div.notion-page-content div[data-block-id]")

        current_section = None
        for block in content_blocks.element_handles():
            block_type = block.get_attribute("data-block-type")
            block_text = block.inner_text().strip()

            if block_type in ["header", "sub_header", "sub_sub_header"]:
                current_section = {"heading": block_text, "content": []}
                structured_data["sections"].append(current_section)
            elif block_type == "text":
                if current_section:
                    current_section["content"].append({"text": block_text})
            elif block_type == "code":
                code_language = block.get_attribute("data-language") or "plaintext"
                code_content = block.inner_text()
                if current_section:
                    current_section["content"].append({"code": {"language": code_language, "content": code_content}})

        # Close the browser
        browser.close()

        return structured_data

# URL to scrape
url = "https://crustdata.notion.site/Crustdata-Discovery-And-Enrichment-API-c66d5236e8ea40df8af114f6d447ab48"

# Run the scraper
result = scrape_notion_page(url)

# Print the structured data in JSON format
print(json.dumps(result, indent=4))

# Optional: Save the result to a file
with open("notion_data.json", "w") as f:
    json.dump(result, f, indent=4)
