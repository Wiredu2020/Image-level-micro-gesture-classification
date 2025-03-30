from .config import CFG
from transformers import BertTokenizer, DistilBertModel, DistilBertConfig, DistilBertTokenizer
from torch import nn
import torch.nn.functional as F
import timm

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        """
        Initializes the image encoder.

        Args:
            model_name (str): Name of the image encoder model.
            pretrained (bool): Whether to use pre-trained weights.
            trainable (bool): Whether the encoder is trainable.
        """
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        """
        Forward pass of the image encoder.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Encoded image features.
        """
        return self.model(x)


class TextEncoder(nn.Module):
    """
    Encodes text to a fixed-size vector.
    """
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        """
        Initializes the text encoder.

        Args:
            model_name (str): Name of the text encoder model.
            pretrained (bool): Whether to use pre-trained weights.
            trainable (bool): Whether the encoder is trainable.
        """
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

        # Using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the text encoder.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for input.

        Returns:
            torch.Tensor: Encoded text features.
        """
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    """
    Projection head module for projecting embeddings.
    """
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        """
        Initializes the projection head.

        Args:
            embedding_dim (int): Dimension of input embeddings.
            projection_dim (int): Dimension of projected embeddings.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        """
        Forward pass of the projection head.

        Args:
            x (torch.Tensor): Input embeddings.

        Returns:
            torch.Tensor: Projected embeddings.
        """
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class RestMGModel(nn.Module):
    """
    RestMGModel a Custom Version of (Contrastive Language-Image Pretraining) model.
    """
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        """
        Initializes the RestMGModel model.

        Args:
            temperature (float): Temperature parameter for contrastive loss.
            image_embedding (int): Dimension of image embeddings.
            text_embedding (int): Dimension of text embeddings.
        """
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        """
        Forward pass of the CLIP model.
 
        Args:
            batch (dict): Dictionary containing image and text inputs.

        Returns:
            tuple: Tuple containing image and text embeddings.
        """
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        return image_embeddings, text_embeddings
    
if __name__ == "__main__":
    pass