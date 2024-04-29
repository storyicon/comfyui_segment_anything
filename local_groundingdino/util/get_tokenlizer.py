from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast
import os

def get_tokenlizer(text_encoder_type: str | dict | os.PathLike) -> AutoTokenizer:
    """Retrieves an AutoTokenizer based on the provided text encoder type, which can be a string, dictionary, or path. If the input is not a string, it extracts the text encoder type accordingly. It raises a ValueError for unknown text encoder types."""
    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        elif os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
            pass
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )
    print("final text_encoder_type: {}".format(text_encoder_type))

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
    return tokenizer


def get_pretrained_language_model(text_encoder_type: str) -> RobertaModel | BertModel:
    """Retrieves a pretrained language model based on the specified text encoder type, either BertModel or RobertaModel. Raises a ValueError for unknown text encoder types."""
    if text_encoder_type == "bert-base-uncased" or (os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type)):
        return BertModel.from_pretrained(text_encoder_type)
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(text_encoder_type)

    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))