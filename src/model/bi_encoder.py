import torch
import torch.nn as nn


class BiEncoder(nn.Module):
    """
    BiEncoder: a simple shallow interaction network for Retrieval-based Chatbot.
    BiEncoder uses context_encoder and response_encoder to
    get context_embedding and response_embedding,
    returns the dot product of them as the matching score.
    """

    def __init__(self, context_encoder, response_encoder):
        super(BiEncoder, self).__init__()
        self._context_encoder = context_encoder
        self._response_encoder = response_encoder

    def get_context_encoder(self):
        return self._context_encoder

    def get_response_encoder(self):
        return self._response_encoder

    def get_embedding(self, input_ids, input_masks, mode=None, norm=False):
        if mode not in ['context', 'response']:
            raise Exception(
                "Can't get %s Embedding, please try in [context, response]!"
                % mode
            )

        encoder = self._context_encoder if mode == 'context' \
            else self._response_encoder
        u_embedding = encoder.get_sentence_embedding(
            input_ids, input_masks)  # [batch, dim]
        if norm:
            u_embedding = u_embedding / torch.norm(u_embedding, dim=-1,
                                                   keepdim=True)  # norm
        return u_embedding

    def forward(self, data=None, context_input_ids=None,
                context_input_masks=None, responses_input_ids=None,
                responses_input_masks=None, cross_dot_product=False,
                norm=False):
        """
        :param data: dict contains(context_input_ids, context_input_masks,
        responses_input_ids, responses_input_masks).
        :param context_input_ids: context processed by tokenizer,
        size: [batch, seq_len].
        :param context_input_masks: context masks processed by tokenizer,
        size: [batch, seq_len].
        :param responses_input_ids: k responses processed by tokenizer,
        size: [batch, k, seq_len].
        :param responses_input_masks: k responses processed by tokenizer,
        size: [batch, k, seq_len].
        :param norm: norm the context and response embedding if norm is True.
        :return: context_response_dot_product: score
        between context and responses, size: [batch, k].
        :param cross_dot_product: if k == 1 then
        return context_response_dot_product size: [batch, batch],
        it means use other response in same batch as negative,
        so 1 context has batch_size responses!
        """
        if data is not None:
            if context_input_ids is None:
                context_input_ids = data['context_input_ids']
            if context_input_masks is None:
                context_input_masks = data['context_input_masks']
            if responses_input_ids is None:
                responses_input_ids = data['responses_input_ids']
            if responses_input_masks is None:
                responses_input_masks = data['responses_input_masks']

        if context_input_ids is None or responses_input_ids is None:
            raise Exception("Lack context_input_ids and responses_input_ids!")

        if len(responses_input_ids.shape) != 3:
            responses_input_ids = responses_input_ids.unsqueeze(1)
            responses_input_masks = responses_input_masks.unsqueeze(1)

        # get context_embedding
        context_vec = self.get_embedding(context_input_ids, context_input_masks,
                                         mode='context', norm=norm)

        # get responses_embedding
        batch_size, res_cnt, seq_length = responses_input_ids.shape
        responses_input_ids = responses_input_ids.view(-1, seq_length)
        responses_input_masks = responses_input_masks.view(-1, seq_length)
        responses_vec = self.get_embedding(responses_input_ids,
                                           responses_input_masks,
                                           mode='response', norm=norm)
        responses_vec = responses_vec.view(batch_size, res_cnt, -1)

        if res_cnt == 1 and cross_dot_product:
            dot_product = torch.mm(context_vec,
                                   responses_vec.squeeze(1).T)  # [batch, batch]
        else:
            dot_product = torch.bmm(
                responses_vec,
                context_vec.unsqueeze(2)
            ).squeeze(2)  # [batch, k]

        return dot_product

    def resize_token_embeddings(self, length):
        self._context_encoder.resize_token_embeddings(length)
        self._response_encoder.resize_token_embeddings(length)
