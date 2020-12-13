from model.bi_encoder import BiEncoder
from module.encoder import BasicBertEncoder


def build_biencoder_model(model_tokenizer, args, aggregation='cls'):
    context_encoder = BasicBertEncoder(param_path=args.pretrain_checkpoint,
                                       aggregation=aggregation)
    response_encoder = BasicBertEncoder(param_path=args.pretrain_checkpoint,
                                        aggregation=aggregation)
    bi_encoder = BiEncoder(context_encoder, response_encoder)
    bi_encoder.resize_token_embeddings(len(model_tokenizer))
    if hasattr(args, "distributed") and args.distributed:
        bi_encoder.cuda(args.local_rank)
    else:
        bi_encoder.to(args.device)
    return bi_encoder
