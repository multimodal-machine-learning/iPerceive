import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from qanet.tvqanet import TVQANet
from tvqa_dataset import TVQADataset, pad_collate, prepare_inputs
from config import BaseOptions, paths
from utils import save_json_pretty, merge_dicts, save_json, loadPreTrainedModel

def find_max_pair(p1, p2):
    """ Find (k1, k2) where k1 <= k2 with the maximum value of p1[k1] * p2[k2]
    Args:
        p1: a list of probablity for start_idx
        p2: a list of probablity for end_idx
    Returns:
        best_span: (st_idx, ed_idx)
        max_value: probability of this pair being correct
    """
    max_val = 0
    best_span = (0, 1)
    argmax_k1 = 0
    for i in range(len(p1)):
        val1 = p1[argmax_k1]
        if val1 < p1[i]:
            argmax_k1 = i
            val1 = p1[i]

        val2 = p2[i]
        if val1 * val2 > max_val:
            best_span = (argmax_k1, i)
            max_val = val1 * val2
    return best_span, float(max_val)


def inference(opt, dset, model):
    dset.set_mode(opt.mode)
    data_loader = DataLoader(dset, batch_size=opt.test_bsz, shuffle=False, collate_fn=pad_collate)

    train_corrects = []
    predictions = dict(ts_answer={}, raw_bbox=[])
    max_len_dict = dict(
        max_sub_l=opt.max_sub_l,
        max_vid_l=opt.max_vid_l,
        max_vcpt_l=opt.max_vcpt_l,
        max_qa_l=opt.max_qa_l,
        max_dc_l=opt.max_dc_l
    )
    for valid_idx, batch in tqdm(enumerate(data_loader)):
        model_inputs, targets, qids = prepare_inputs(batch, max_len_dict=max_len_dict, device=opt.device)

        inference_outputs, _ = model(model_inputs)

        # predicted answers
        pred_ids = inference_outputs.data.max(1)[1]

        train_corrects += pred_ids.eq(targets.data).tolist()
        train_acc = sum(train_corrects) / float(len(train_corrects))
        print(train_corrects)

    print("Idx {:02d} [Train] acc {:.4f}".format(valid_idx, train_acc))

        # predicted regions
        # if inference_outputs["att_predictions"]:
        #     predictions["raw_bbox"] += inference_outputs["att_predictions"]
        #
        # temporal_predictions = inference_outputs["t_scores"]
        # for qid, pred_a_idx, temporal_score_st, temporal_score_ed, img_indices in \
        #         zip(qids, pred_ids.tolist(),
        #             temporal_predictions[:, :, :, 0],
        #             temporal_predictions[:, :, :, 1],
        #             model_inputs["image_indices"]):
        #     offset = (img_indices[0] % 6) / 3
        #     (st, ed), _ = find_max_pair(temporal_score_st[pred_a_idx].cpu().numpy().tolist(),
        #                                 temporal_score_ed[pred_a_idx].cpu().numpy().tolist())
        #     # [[st, ed], pred_ans_idx], note that [st, ed] is associated with the predicted answer.
        #     predictions["ts_answer"][str(qid)] = [[st * 2 + offset, (ed + 1) * 2 + offset], int(pred_a_idx)]

    return predictions


def main_inference():
    print("Loading config...")
    opt = BaseOptions().parse()
    print("Loading dataset...")
    dset = TVQADataset(opt, paths)
    print("Loading model...")
    model = TVQANet(opt)

    device = torch.device("cuda:0" if opt.device != '-2' and torch.cuda.is_available() else "cpu")

    # if specified, use opt.device else use the better of whats available (gpu > cpu)
    #model.to(opt.device if opt.device != '-2' else device)

    cudnn.benchmark = True

    # load pre-trained model if it exists
    loadPreTrainedModel(model=model, modelPath=paths["pretrained_model"])

    model.eval()
    model.inference_mode = True
    torch.set_grad_enabled(False)
    print("Evaluation Starts:\n")
    predictions = inference(opt, dset, model)
    print("predictions {}".format(predictions.keys()))
    pred_path = paths["pretrained_model"].replace("best_valid.pth", "{}_inference_predictions.json".format(opt.mode))
    save_json(predictions, pred_path)


if __name__ == "__main__":
    main_inference()
