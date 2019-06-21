"""
Originally based on https://github.com/zihangdai/xlnet/blob/master/prepro_utils.py
"""

from typing import List, Union

import sentencepiece as spm


SPIECE_UNDERLINE = "▁"


class TokenIndexer:
    """Encode tokens and decode token ids using sentencepiece."""
    def __init__(self, spm_model_path: str):
        self.spm_model = spm.SentencePieceProcessor()
        self.spm_model.load(spm_model_path)

    def encode(self, text: str, sample: bool=False) -> List[str]:
        return self._encode_pieces(text, sample)

    def _encode_pieces(self, text: str, sample: bool=False) -> List[str]:
        if not sample:
            pieces = self.spm_model.EncodeAsPieces(text)
        else:
            pieces = self.spm_model.SampleEncodeAsPieces(text, 64, 0.1)

        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == "," and piece[-2].isdigit():
                cur_pieces = self.spm_model.EncodeAsPieces(
                    piece[:-1].replace(SPIECE_UNDERLINE, ""))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)
        return new_pieces

    def decode(self, encoded_ids: List[int], return_piece: bool=False) -> Union[str, List[str]]:
        """Decode encoded ids. If `return_piece` is True, return pieces (List[str])."""
        if return_piece:
            return [self.spm_model.IdToPiece(encoded_id) for encoded_id in encoded_ids]
        return self.spm_model.DecodeIds(encoded_ids)
