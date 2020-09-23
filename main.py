import sys
import cv2 as cv
from imutils.object_detection import non_max_suppression
import numpy as np
from PIL import Image
import pytesseract
import pandas as pd
import spacy

class ImageProcessor:
	def __init__(self, image):
		self.img = cv.imread(image)
		self.books = []
		self.titles_and_authors = {}
		self.num_books = 0
		self.df = None


	# Isolate individual books and create a list of cropped images
	def isolate_books(self):
		copy = self.img.copy()

		y_max = self.img.shape[0]
		x_max = self.img.shape[1]

		prev = []
		prev_1 = []
		y = y_max // 2

		low = []
		high = []

		# Go through the image at two different heights
		for height in ["low", "high"]:
			if height == "low":
				y = y_max // 3
			elif height == "high":
				y = (y_max * 2) // 3

			colors = []

			# Go from left to right across the image
			for x in range(x_max):
				current = self.img[y][x]
				if colors == []:
					colors.append((current, x))

				# If the color is "different enough" from the previous, add it to the list
				else:
					if pixel_difference(colors[len(colors) - 1][0], current) < 50:
						colors[len(colors) - 1] = (current, colors[len(colors) - 1][1])
						continue
					else:
						colors.append((current, x))

			# Add a boolean attribute to the list for later use
			colors_new = [(color[0], color[1], True) for color in colors]

			for i in range(len(colors)):
				temp = colors[i][0]

				repeat = 0
				index = i

				for elem in colors[i+1:]:
					index += 1
					if pixel_difference(elem[0], temp) < 50:
						repeat = index
						break

				# If the color is repeated further to the right, but still close by
				# Then remove everything in between as it is most likely the same book
				# Instead of actually removing it, the third tuple value of is just set to False
				if repeat > i and repeat - i < 4:
					for x in range(i, repeat + 1):
						colors_new[x] = (colors_new[x][0], colors_new[x][1], False)

			if height == "low":
				low = colors_new
			elif height == "high":
				high = colors_new

		# Find color changes at similar x coordinates from the two heights
		# Then crop the image at that point
		prev_line = []
		for color in low:
			if color[2] == True:
				add_line = False
				for other_color in high:
					if abs(color[1] - other_color[1]) < 40:
						add_line = True
						break
				if add_line:
					if prev_line == []:
						prev_line = color
					elif color[1] - prev_line[1] < 140:
						continue
					else:
						book = copy.copy()
						crop_book = book[0:y_max-1, prev_line[1]:color[1]]
						prev_line = color
						self.books.append(crop_book)

		self.num_books = len(self.books)


	# Take each book image and extract the text from it
	def text_extraction(self):
		for count, book in enumerate(self.books):
			sections = []
			flipped = False

			# Convert the image to binary using thresholding and then dilate it
			gray = cv.cvtColor(book, cv.COLOR_BGR2GRAY) 
	  
			ret, thresh1 = cv.threshold(gray, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV) 
			 
			rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (12, 12))
			  
			dilation = cv.dilate(thresh1, rect_kernel, iterations = 1)
			 
			# Find the contours of the image
			contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

			# If there aren't many contours, invert the image
			if len(contours) < 5:
				dilation = cv.bitwise_not(dilation)
				contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
				flipped = True

			book2 = book.copy()

			ordered_contours = []

			# Only add contours that are small enough and have a small enough width/height to a new list
			for cnt in contours:
				area = cv.contourArea(cnt)
				if area > (book.shape[0] * book.shape[1] * 0.05): continue

				x, y, w, h = cv.boundingRect(cnt)

				if w < 60 or h < 20: continue

				ordered_contours.append((x, y, w, h, area))

			# Sort the list by y value
			ordered_contours = sorted(ordered_contours, key=lambda x: x[1])
			
			prev_x = -1
			prev_y = -1
			prev_w = -1
			prev_h = -1

			full_x = -1
			full_y = -1
			full_w = -1
			full_h = -1

			# Loop through the ordered contour list
			for index, cnt in enumerate(ordered_contours):
				x, y, w, h, area = cnt

				if prev_x == -1 and prev_y == -1 and prev_w == -1 and prev_h == -1:
					prev_x = x
					prev_y = y
					prev_w = w
					prev_h = h

					full_x = x
					full_y = y
					full_w = w
					full_h = h
					continue

				# If the widths of the two MBRs are similar and the distance between the two is small
				# Then update the MBR to encompass both
				if abs(w - prev_w) < 30 and abs(y - prev_y - prev_h) / w < 1:
					if full_x < x:
						full_w = abs(x - full_x) + w
					full_h = abs(full_y - y) + h
					full_x = prev_x if prev_x < full_x else full_x
					
					# If this is the last contour in the list, then make sure to crop based on the new expanded MBR
					if index == len(ordered_contours) - 1:
						# If the text is the right shape, i.e. longer than it is high then crop
						if full_w < full_h / 3:
							# rect = cv.rectangle(book2, (full_x, full_y), (full_x + full_w, full_y + full_h), (255, 0, 0), 2)

							# Add 40 to each side so that there is enough whitespace for tesseract to work
							crop_y_start = full_y - 40 if full_y - 40 > 0 else 5
							crop_y_end = full_y + full_h + 40 if full_y + full_h + 40 < book2.shape[0] else book2.shape[0] - 5

							crop_x_start = full_x - 40 if full_x - 40 > 0 else 5
							crop_x_end = full_x + full_w + 40 if full_x + full_w + 40 < book2.shape[1] else book2.shape[1] - 5

							# cropped = book2[full_y - 20:full_y + full_h + 20, full_x - 20:full_x + full_w + 20]
							cropped = book2[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
							sections.append(cropped)

					prev_x = x
					prev_y = y
					prev_w = w
					prev_h = h

				# If the current contour doesn't match the previous then check if the previous MBR should be cropped
				else:
					# If the text is the right shape, crop it
					if full_w < full_h / 3:

						# Add 40 to each side so that there is enough whitespace for tesseract to work
					    crop_y_start = full_y - 40 if full_y - 40 > 0 else 5
					    crop_y_end = full_y + full_h + 40 if full_y + full_h + 40 < book2.shape[0] else book2.shape[0] - 5

					    crop_x_start = full_x - 40 if full_x - 40 > 0 else 5
					    crop_x_end = full_x + full_w + 40 if full_x + full_w + 40 < book2.shape[1] else book2.shape[1] - 5

					    cropped = book2[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
					    sections.append(cropped)

					prev_x = x
					prev_y = y
					prev_w = w
					prev_h = h
					full_x = x
					full_y = y
					full_w = w
					full_h = h

			# cv.imwrite('text_dection.jpg', book2)
			if count not in self.titles_and_authors:
				self.titles_and_authors[count] = []

			# For each section, pass it to tesseract to conver to text, and output it
			for section in sections:
				gray = cv.cvtColor(section, cv.COLOR_BGR2GRAY)
				gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
				if flipped:
					gray = cv.bitwise_not(gray)
				cv.imwrite('Images/test_grayscale.jpg', gray)

				text = pytesseract.image_to_string(Image.open('Images/test_grayscale.jpg'))
				text = text.replace('\n\x0c', '')
				if text: text = text.title()

				self.titles_and_authors[count].append(text)

	# Turn text data into pandas dataframe
	# Use NLP to determine title vs author
	def create_dataframe(self):
		self.df = pd.DataFrame.from_dict(self.titles_and_authors, orient='index', columns = ["Author", "Title"])

		nlp = spacy.load("en_core_web_sm")

		for index, row in self.df.iterrows():
			author = row['Author']
			title = row['Title']
			author_tokens = []
			title_tokens = []

			if author:
				author_nlp = nlp(author.title())
				author_tokens = [token for token in author_nlp if token.ent_type_ == 'PERSON']

			if title:
				title_nlp = nlp(title.title())
				title_tokens = [token for token in title_nlp if token.ent_type_ == 'PERSON']

			if len(author_tokens) < len(title_tokens) or len(author_tokens) == 0 and title == None:
				self.df.at[index, "Title"] = author
				self.df.at[index, "Author"] = title


# Find distance between two pixels in RGB format
def pixel_difference(a, b):
	a = a.astype(int)
	b = b.astype(int)
	diff = abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
	return diff


if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Usage: python main.py <image_file>")
		exit()

	image = sys.argv[1]

	img_processor = ImageProcessor(image)
	img_processor.isolate_books()
	img_processor.text_extraction()
	img_processor.create_dataframe()

	print(img_processor.df)

