# MiniBench Noto Sans CJK SC subsets

The two `.otf` files in this directory are deterministic glyph subsets of the
official Noto Sans CJK SC OpenType fonts. They are distributed under the SIL
Open Font License 1.1; see `licenses/NotoSansCJK-OFL-1.1.txt`.

Sources downloaded from the `notofonts/noto-cjk` GitHub repository:

- `Sans/SubsetOTF/SC/NotoSansCJKsc-Regular.otf`
  - source SHA256: `2C76254F6FC379FDDFCE0A7E84FB5385BB135D3E399294F6EEB6680D0365B74B`
- `Sans/SubsetOTF/SC/NotoSansCJKsc-Bold.otf`
  - source SHA256: `B5F0D1A190A7F9B43C310A8850630AF12553DF32C4C050543F9059732D9B4C0A`

Generated files:

- `NotoSansCJKsc-MiniBench-Regular.otf`
  - SHA256: `55B4F67B959BCDD092F810A2FEE2D6E71A7E138D8AB059F491B800E933513F24`
- `NotoSansCJKsc-MiniBench-Bold.otf`
  - SHA256: `AD0E7D6FBCADBA0907804611C9FF7D53F34405A0292BCA2F6C33C48E44BCF764`

The subset contains ASCII (`U+0020-007E`) plus the Chinese glyphs used by the
Xiangqi renderer and gallery, including 帅、将、仕、士、相、象、马、馬、车、車、
炮、砲、兵、卒、红、黑、方、棋、局、目、标、规、则、难、度、最、佳、走、法、
历、史、回、合、检、索、全、部、无、未、指、定、简、体、中、文、图、像.

Subsets were created with fonttools `pyftsubset`, retaining layout features and
notdef glyphs and using canonical glyph order.
