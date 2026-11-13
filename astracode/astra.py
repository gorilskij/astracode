"""Astra's own use of Astracode: a member number in a one-ring code.

The format (spec.py) carries a string of bits and gives it no meaning. This
module is one convention on top of it, the one Astra's apps follow:

  - A payload of exactly NUMBER_BITS = 38 bits is an Astra number. That is
    the most one ring holds, so an Astra number is always a one-ring code,
    and since 38 bits is not a whole number of bytes it cannot be mistaken
    for a byte string.
  - Read as an integer, the 38 bits are metadata * 10^10 + payload: a
    10-digit payload (0 .. 9 999 999 999) and a metadata value 0 .. 26 saying
    what the payload is.
  - A member ID is 12 digits: the 10-digit payload followed by the payload
    mod 97. Only the first 10 are stored; the last two are recomputed when
    reading.
"""
from . import spec

NUMBER_BITS = 38
PAYLOAD_LIMIT = 10 ** 10                               # payloads are 0 .. PAYLOAD_LIMIT - 1
METADATA_LIMIT = (1 << NUMBER_BITS) // PAYLOAD_LIMIT   # 27: metadata is 0 .. 26

assert spec.capacity_bits(1) == NUMBER_BITS


def pack(metadata, payload):
    """Metadata 0..26 and a 10-digit payload -> the 38-bit value of an Astra number."""
    if not 0 <= metadata < METADATA_LIMIT:
        raise ValueError("metadata must be 0..%d" % (METADATA_LIMIT - 1))
    if not 0 <= payload < PAYLOAD_LIMIT:
        raise ValueError("payload must have at most 10 digits")
    return metadata * PAYLOAD_LIMIT + payload


def unpack(value, nbits):
    """A code's payload -> (metadata, payload) if it is an Astra number, else None."""
    if nbits != NUMBER_BITS or value >= METADATA_LIMIT * PAYLOAD_LIMIT:
        return None
    return divmod(value, PAYLOAD_LIMIT)


def member_id(payload):
    """The 12-digit member ID for a 10-digit payload: the payload, then payload mod 97."""
    return "%010d%02d" % (payload, payload % 97)


def payload_of_member_id(text):
    """12-digit member ID -> its 10-digit payload; ValueError if the last two digits are wrong."""
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) != 12 or int(digits[:10]) % 97 != int(digits[10:]):
        raise ValueError("not a valid member ID")
    return int(digits[:10])


def encode(payload, metadata=0):
    """The symbols of the one-ring code for an Astra number, as [ring 0's 12 symbols]."""
    return spec.encode_bits(pack(metadata, payload), NUMBER_BITS)
