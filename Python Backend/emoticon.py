import re
from emot.emo_unicode import UNICODE_EMO, EMOTICONS


# Function for converting emojis into word
def convert_emojis(text):
    for emot in UNICODE_EMO:
        text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",", "").replace(":", "").split()))
    return text


# Function for converting emoticons into word
def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u'(' + emot + ')', "_".join(EMOTICONS[emot].replace(",", "").split()), text)
    return text


# Emoji Convert
text = "Hilarious 😂. The feeling of making a sale 😎, The feeling of actually fulfilling orders 😒"
text_emoji = convert_emojis(text)
print('TEXT EMOJI CONVERT')
print(text_emoji)

# Emoticon Convert
text = "Hello :-) :P :o :| :( :x"
text_emoticon = convert_emoticons(text)
print('TEXT EMOTICON CONVERT')
print(text_emoticon)
