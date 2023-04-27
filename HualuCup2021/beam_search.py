import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
import operator


SOS_token = 2
EOS_token = 3

class BeamSearchNode(object):
    def __init__(self, previousNode, wordId, prev_wordids, logProb, length):
        self.prevNode = previousNode
        self.wordid = wordId
        self.prev_wordids = prev_wordids
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


def beam_decode(model1, model2, model3, image, device):
    beam_width = 5
    topk = 1
    decoded_batch = []

    for idx in range(1):

        decoder_input = torch.LongTensor([[SOS_token]]).to(device)

        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        node = BeamSearchNode(None, decoder_input, decoder_input, 0, 1)
        nodes = PriorityQueue()

        nodes.put((-node.eval(), node))
        qsize = 1


        while True:
            if qsize > 2000: break

            try: 
                score, n = nodes.get()
                decoder_input = n.prev_wordids

                if n.wordid.item() == EOS_token and n.prevNode != None:
                    endnodes.append((score, n))

                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                decoder_output1 = model1.caption_step(image, decoder_input)
                decoder_output2 = model2.caption_step(image, decoder_input)
                decoder_output3 = model3.caption_step(image, decoder_input)

                decoder_output = 0.4 * decoder_output1 + 0.4 * decoder_output2 + 0.2 * decoder_output3

                log_prob, indexes = torch.topk(decoder_output, beam_width)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()

                    prev_wordids = torch.cat((decoder_input, decoded_t), 1)

                    node = BeamSearchNode(n, decoded_t, prev_wordids, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))

                qsize += len(nextnodes) - 1
            except:
                return decoded_batch

        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch

