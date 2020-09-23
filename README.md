# Bookshelf Image Processor
Given an image of a bookshelf as input, returns a dataframe of titles and authors using image processing and text extraction.

## Setup
Initialize a virtual environment.

    virtualenv venv
    
Activate the virtual environment.

    source venv/bin/activate

Install dependencies.

    pip install -r requirements.txt

## Usage
    python main.py <image_file>

Use Images/test.jpg for a test image.

## Description
Input should be an image of a bookshelf similar to the example below:

<img src="https://github.com/andrewjones4/BookshelfImageProcessing/blob/master/Images/test.jpg" width="450" height="600">

The program then splits the image into individual books, based on an algorithm that determines color changes at different y-coordinate values.

Examples of individual books (rotated 90 degrees):

<img src="https://github.com/andrewjones4/BookshelfImageProcessing/blob/master/Images/book1.png" width="600" height="60">
<img src="https://github.com/andrewjones4/BookshelfImageProcessing/blob/master/Images/book2.png" width="600" height="60">

Then, using contour detection, the program finds minimum bounding rectangles (MBRs) around individual characters.

<img src="https://github.com/andrewjones4/BookshelfImageProcessing/blob/master/Images/character_MBR_example.png" width="600" height="60">

Using size, color, and width, the program then determines which characters should be grouped together, placing a new MBR around the whole title/author.

<img src="https://github.com/andrewjones4/BookshelfImageProcessing/blob/master/Images/title_author_MBR_example.png" width="600" height="60">

Then, using the pytesseract library, the program extracts the text from these MBRs, storing the data in a pandas DataFrame.

The final step is to use a little natural language processing to help determine what is the author and what is the title, with named entity recognition, then outputting the updated DataFrame to the terminal.
