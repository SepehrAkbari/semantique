# semantique

semantique is a semi-intelligent natural language processing system that predicts movie ratings based on written reviews. By quantifying subjective language, semantique maps free-form film critiques into a 1–5 star rating system using deep learning.

This project explores data preprocessing, model design with PyTorch, and deployment of the trained neural network through a minimal Flask web application. The final app allows users to input any English-language movie review and receive a predicted rating displayed visually using 5 stars.

## App Demo

Inputting a review:

![Input](App/Demo/page1.jpeg)

Predicted rating:

![Predictions](App/Demo/page2.jpeg)

## Data

The dataset used for training is the IMDb Large Movie Review Dataset, which contains 50,000 reviews labeled with ratings from 1 to 10.

In this project:

- Only labeled reviews were used.

- Ratings were normalized and mapped back to a 1–5 star scale during inference.

- Data was preprocessed with tokenization, vocabulary building, and padding for model input.

## Usage

To run the project, first clone the repository:

```bash
git clone https://github.com/SepehrAkbari/semantique
cd semantique
```

Create a virtual environment (optional but recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r setup/requirements.txt
```

To launch the Flask app:

```bash
flask run
```

To explore the analysis, run the Jupyter Notebook located in the `notebook` folder:

```bash
jupyter notebook notebook/main.ipynb
```

## Approach

- **Preprocessing:**

    - Cleaned reviews using custom tokenization

    - Built a vocabulary of the most frequent words (capped at 20,000)

    - Encoded and padded sequences for input into the neural network

- **Model Architecture:**

    - Embedding layer to learn word representations

    - LSTM for sequence modeling with dropout

    - Fully connected layer with a scalar output in the range 1–10, rounded to produce half-star ratings

- **Training:**

    - Used MSE loss for regression

    - Early stopping and validation monitoring were employed

    - Final model achieved strong performance on unseen test reviews

- **Deployment:**

    - The trained model and vocabulary are loaded at runtime

    - A Flask app accepts reviews, runs predictions, and displays the results as a 5-star visual scale

## Contributing

Feel free to fork the repository, open issues, or submit pull requests for enhancements. If you find a bug or want to suggest a new feature, start a discussion!

## License

This project is licensed under the [GNU General Public License (GPL)](/LICENSE).
