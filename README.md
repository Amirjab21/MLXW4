This week, we built a decoder model that generates captions for images. I trained using the Flickr30k dataset and used CLIP as an image encoder. 

I also used CLIPs tokenizer for text but I used my own embeddings for each text token as this performed better than using CLIPs.

I experimented with weighted decay, padding, sequence length, batch size and more to find the optimal model.

Demo available at https://caption-generator-flickr.streamlit.app/
