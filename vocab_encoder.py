"""
Issue #2: Vocabulary / Charset Encoder
GSoC 2026 - RenAIssance OCR Project
Converts characters <-> indices for CTC training
"""

class VocabEncoder:
    """
    Character-level vocabulary encoder for CTC-based OCR.
    
    Index 0 is reserved for CTC blank token.
    All characters start from index 1.
    """

    def __init__(self, charset: str = None):
        """
        Args:
            charset: string of all valid characters.
                     Defaults to digits + lowercase + uppercase + common symbols.
        """
        if charset is None:
            charset = (
                "0123456789"
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                " !\"#&'()*+,-./:;?@[\\]^_`{|}~"
            )

        self.charset = charset
        self.blank_token = "<BLANK>"
        self.blank_index = 0

        # char -> index (starts at 1, 0 reserved for blank)
        self.char2idx = {ch: idx + 1 for idx, ch in enumerate(charset)}

        # index -> char
        self.idx2char = {idx + 1: ch for idx, ch in enumerate(charset)}
        self.idx2char[0] = self.blank_token

    def encode(self, text: str) -> list:
        """
        Convert a text string to list of indices.

        Args:
            text: input string e.g. "Hello"
        Returns:
            list of ints e.g. [33, 60, 57, 57, 60]
        Raises:
            ValueError: if a character is not in charset
        """
        indices = []
        for ch in text:
            if ch not in self.char2idx:
                raise ValueError(
                    f"Character '{ch}' not in charset. "
                    f"Add it to charset or filter your dataset."
                )
            indices.append(self.char2idx[ch])
        return indices

    def decode(self, indices: list, remove_blank: bool = True) -> str:
        """
        Convert list of indices back to string.

        Args:
            indices: list of ints
            remove_blank: if True, removes blank tokens (index 0)
        Returns:
            decoded string
        """
        chars = []
        for idx in indices:
            if idx == self.blank_index and remove_blank:
                continue
            chars.append(self.idx2char.get(idx, "?"))
        return "".join(chars)

    def ctc_decode(self, indices: list) -> str:
        """
        CTC greedy decode: collapse repeated chars, then remove blanks.

        Args:
            indices: raw output indices from model
        Returns:
            clean decoded string
        """
        # Step 1: collapse consecutive duplicates
        collapsed = []
        prev = None
        for idx in indices:
            if idx != prev:
                collapsed.append(idx)
                prev = idx

        # Step 2: remove blank tokens
        return self.decode(collapsed, remove_blank=True)

    def __len__(self):
        """Returns vocab size including blank token."""
        return len(self.charset) + 1  # +1 for blank

    @property
    def num_classes(self):
        """Alias for CTC output layer size."""
        return len(self)

    def __repr__(self):
        return (
            f"VocabEncoder("
            f"charset_size={len(self.charset)}, "
            f"num_classes={self.num_classes}, "
            f"blank_index={self.blank_index})"
        )


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    enc = VocabEncoder()
    print(enc)
    print(f"Vocab size (with blank): {enc.num_classes}")

    # Encode test
    text = "Hello 2026"
    indices = enc.encode(text)
    print(f"\nOriginal : {text}")
    print(f"Encoded  : {indices}")

    # Decode test
    decoded = enc.decode(indices)
    print(f"Decoded  : {decoded}")
    assert decoded == text, "Encode-decode mismatch!"
    print("✅ Encode/Decode test passed!")

    # CTC decode test (simulates model output with blanks & repeats)
    raw_ctc = [0, 33, 33, 60, 0, 57, 57, 0, 57, 60, 0]
    print(f"\nRaw CTC  : {raw_ctc}")
    print(f"CTC out  : '{enc.ctc_decode(raw_ctc)}'")
    print("✅ CTC decode test passed!")

    # Edge case: unknown character
    try:
        enc.encode("Hello 😀")
    except ValueError as e:
        print(f"\n✅ Unknown char caught: {e}")
