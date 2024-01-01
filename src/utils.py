import argparse
import pickle
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training_data', type=str, default='/kaggle/input/nrms-dt/training_data.pkl',
                        help='Input training data path')

    parser.add_argument('--validation_data', type=str,
                        default='/kaggle/input/nrms-dt/validation_data.pkl',
                        help='Input validation data path')

    parser.add_argument('--evaluation_data', type=str, default='/kaggle/input/nrms-dt/evaluation_data.pkl',
                        help='Input validation data path')

    parser.add_argument('--category_id', type=str, default='/kaggle/input/nrms-dt/category_id.pkl',
                        help='category_id dictionary')

    parser.add_argument('--subcategory_id', type=str, default='/kaggle/input/nrms-dt/subcategory_id.pkl',
                        help='subcategory_id dictionary')

    parser.add_argument('--TitleWord_id', type=str, default='/kaggle/input/nrms-dt/TitleWord_id.pkl',
                        help='TitleWord_id dictionary')

    parser.add_argument('--AbstractWord_id', type=str, default='/kaggle/input/nrms-dt/AbstractWord_id.pkl',
                        help='AbstractWord_id.pkl')

    parser.add_argument('--EntityID_id', type=str, default='/kaggle/input/nrms-dt/entityID_id.pkl',
                        help='EntityID_id dictionary')

    parser.add_argument('--TitleWordId_embeddings', type=str, default='/kaggle/input/nrms-dt/TitleWordId_embeddings.npy',
                        help='TitleWordId_embeddings matrix')

    parser.add_argument('--AbstractWordId_embeddings', type=str, default='/kaggle/input/nrms-dt/AbstractWordId_embeddings.npy',
                        help='AbstractWordId_embeddings matrix')

    parser.add_argument('--EntityId_embeddings', type=str, default='/kaggle/input/nrms-dt/EntityId_embeddings.npy',
                        help='EntityId_embeddings matrix')

    parser.add_argument('--newsID_categoryID', type=str, default='/kaggle/input/nrms-dt/newsID_categoryID.pkl',
                        help='newsID_categoryID dictionary')

    parser.add_argument('--newsID_subcategoryID', type=str, default='/kaggle/input/nrms-dt/newsID_subcategoryID.pkl',
                        help='newsID_subcategoryID dictionary')

    parser.add_argument('--newsID_TitleWordID', type=str, default='/kaggle/input/nrms-dt/newsID_TitleWordID.pkl',
                        help='newsID_TitleWordId dictionary')

    parser.add_argument('--newsID_AbstractWordID', type=str, default='/kaggle/input/nrms-dt/newsID_AbstractWordID.pkl',
                        help='newsID_AbstractWordId dictionary')

    parser.add_argument('--newsID_titleEntityId_conf', type=str, default='/kaggle/input/nrms-dt/newsID_titleEntityId_conf.pkl',
                        help='newsID_titleEntityId dictionary')

    parser.add_argument('--newsID_abstractEntityId_conf', type=str, default='/kaggle/input/nrms-dt/newsID_abstractEntityId_conf.pkl',
                        help='newsID_abstractEntityId dictionary')

    parser.add_argument('--num_head_text', type=int, default=4,
                        help='num of attention heads for text')

    parser.add_argument('--num_head_entity', type=int, default=4,
                        help='num of attention heads for entity')

    parser.add_argument('--text_attn_vector_size', type=int, default=64,
                        help='dim of attn vector for encoding title/abstract')

    parser.add_argument('--entity_attn_vector_size', type=int, default=32,
                        help='dim of attn vector for encoding title entities/abstract entities')

    parser.add_argument('--news_final_attn_vector_size', type=int, default=24,
                        help='dim of attn vector for encoding a news')

    parser.add_argument('--news_final_embed_size', type=int, default=24,
                        help='the size of a piece of embeded news')

    parser.add_argument('--history_num_head', type=int, default=4,
                        help='num of attention heads for user histories')

    parser.add_argument('--history_attn_vector_size', type=int, default=16,
                        help='dim of attn vector for encoding history')

    parser.add_argument('--recent_num_head', type=int, default=4,
                        help='num of attention heads for user recent behavior')

    parser.add_argument('--recent_attn_vector_size', type=int, default=16,
                        help='dim of attn vector for encoding recent reading behaviors')

    parser.add_argument('--final_attn_vector_size', type=int, default=16,
                        help='dim of attn vector for encoding a news')

    parser.add_argument('--batch_size', type=int, default=6,
                        help='batch size of training')

    parser.add_argument('--model_name', type=str, default='/kaggle/working/NRMS_new.pkl', help='the name of trained model')

    parser.add_argument('--pack_loss', type=str, default='/kaggle/working/pack_loss.pkl', help='the name of the loss file')

    parser.add_argument('--ranking_name', type=str, default='/kaggle/working/NRMS_new-for-MIND/prediction.txt', help='the name of the prediction file')

    return parser.parse_args()
