"""Surface realisation.

Design rules that make the probing result defensible:

1. Every attribute value has >= 10 distinct paraphrases, most of which never
   use the literal label word ("matte" is realised as "a chalk-flat surface",
   "a dead-flat look that caught no light", ...).  A probe that only does
   keyword detection cannot succeed.
2. Phrase banks *and* sentence skeletons are partitioned by index into
   train / val / test.  A test narrative therefore contains no wording the
   probe has ever seen, only unseen wordings of the same underlying state.
3. Colour phrasings are entity-agnostic, so "which region is this colour bound
   to" cannot be read off the colour wording itself.
"""
from __future__ import annotations

import re

SPLITS = ("train", "val", "test")
# index -> split.  6 / 2 / 2 of every bank of ten.
_SPLIT_OF_INDEX = ["train"] * 6 + ["val"] * 2 + ["test"] * 2


def bank_for_split(bank: list[str], split: str) -> list[str]:
    return [p for i, p in enumerate(bank) if _SPLIT_OF_INDEX[i] == split]


ENTITY_PHRASES: dict[str, list[str]] = {
    "lips": [
        "her lips", "her mouth", "the lip area", "her pout",
        "the centre of her mouth", "her upper and lower lip",
        "the lip line and inward", "the contour of her mouth",
        "the soft skin of her lips", "the surface of her lips",
    ],
    "eyes": [
        "her eyelids", "her eyes", "the lid space", "her mobile lid",
        "the crease and lid", "the skin above her lashes",
        "her eye area", "the lids and outer corner",
        "the socket line", "the hooded part of her lid",
    ],
    "cheeks": [
        "her cheeks", "the apples of her cheeks", "her cheekbones",
        "the high points of her face", "the round of her cheek",
        "her cheek area", "the sides of her face",
        "the rise beneath her eyes", "her upper cheek", "the outer cheek",
    ],
    "skin": [
        "her complexion", "her skin", "the canvas of her face", "her base",
        "the whole of her face", "her overall skin tone",
        "the surface of her face", "her all-over base",
        "the broad planes of her face", "her general complexion",
    ],
}

COLOR_PHRASES: dict[str, list[str]] = {
    "nude": [
        "a soft nude tone", "a bare-skin beige", "a neutral flesh-toned shade",
        "something barely-there and neutral", "a pale buff colour",
        "a skin-matching tone", "a muted oatmeal shade",
        "a quiet, colourless-looking beige", "a shade that read like skin",
        "a naked, undertoned neutral",
    ],
    "pink": [
        "a soft pink", "a rosy tone", "a cool blush shade",
        "a petal-toned colour", "a light rose", "a candy-toned shade",
        "a fresh rosewater tone", "a colour between blush and bubblegum",
        "a pale peony tone", "a sweet, girlish rose",
    ],
    "red": [
        "a bold red", "a deep crimson", "a true scarlet",
        "a classic cherry shade", "a fire-engine tone", "a rich vermilion",
        "a bright poppy colour", "a saturated, letterbox-bright shade",
        "a stop-sign bright tone", "a dark ruby colour",
    ],
    "brown": [
        "a warm brown", "a chocolate tone", "a cocoa shade",
        "a deep coffee colour", "a toasted caramel tone",
        "an earthy chestnut shade", "a bronzed, sun-baked colour",
        "a dark, roasted-bean tone", "a tobacco-toned shade",
        "a burnt-cinnamon colour",
    ],
}

FINISH_PHRASES: dict[str, list[str]] = {
    "matte": [
        "a matte finish", "a flat, no-shine surface",
        "a completely shine-free look", "a velvety dry finish",
        "a blotted, gloss-free surface", "a powdery light-absorbing finish",
        "a chalk-flat surface", "a dead-flat look that caught no light",
        "a soft un-reflective finish", "a fully dried-down, non-wet look",
    ],
    "dewy": [
        "a dewy finish", "a wet-looking shine",
        "a glossy light-catching surface", "a glistening, damp-looking finish",
        "a fresh just-splashed sheen", "a slick reflective surface",
        "a glassy high-shine finish", "a lit-from-within glow",
        "a juicy liquid-looking gleam", "a still-damp-seeming shine",
    ],
    "satin": [
        "a satin finish", "a soft low-key sheen",
        "a finish between flat and shiny", "a muted lustre",
        "a gentle silky glow", "a not-quite-matte, not-quite-wet surface",
        "a subtle powdery shine", "a half-shine, half-blur finish",
        "a smooth semi-gloss surface", "an understated silk-like surface",
    ],
}

COVERAGE_PHRASES: dict[str, list[str]] = {
    "light": [
        "a sheer wash", "a hint of colour", "a thin, translucent layer",
        "a barely-there veil", "a single whisper-thin pass",
        "a see-through film", "a soft diffused trace",
        "the faintest possible amount", "a transparent stain",
        "a light, skin-showing layer",
    ],
    "full": [
        "full opaque coverage", "a solid built-up layer",
        "a dense complete layer", "total blocked-out coverage",
        "an opaque saturated layer", "a heavy, colour-total layer",
        "a thick, nothing-showing-through layer",
        "a full-strength undiluted layer", "maximum coverage",
        "a packed-on, solid layer",
    ],
}

TEMPLATES: dict[str, list[str]] = {
    "APPLY": [
        "She swept {col} across {ent}, building it to {cov} with {fin}.",
        "Working slowly, she brought {col} onto {ent} as {cov}, finishing in {fin}.",
        "{Col} went onto {ent} in {cov}, and it settled into {fin}.",
        "She pressed {col} into {ent}; the result was {cov} with {fin}.",
        "Onto {ent} she blended {col} as {cov}, and it dried to {fin}.",
        "A brush loaded with {col} moved over {ent}, leaving {cov} and {fin}.",
        "She covered {ent} in {col}, taking it to {cov} until it read as {fin}.",
        "{Ent} took {col} next, worked up to {cov} and {fin} once it set.",
        "With a few strokes she put {col} on {ent} as {cov}, ending in {fin}.",
        "She laid {col} over {ent} in {cov}; it came out with {fin}.",
    ],
    "BLOT": [
        "She pressed a tissue to {ent} until every trace of shine was gone.",
        "A quick dusting of powder over {ent} killed the light completely.",
        "She blotted {ent}, taking the surface flat.",
        "Whatever gloss had been on {ent} was absorbed away.",
        "She dabbed at {ent} with folded paper until it stopped reflecting.",
        "The shine on {ent} was pressed out entirely.",
        "She set {ent} with powder, and it went velvety and dry.",
        "{Ent} lost its sheen under a light press of tissue.",
        "She worked {ent} until it looked chalk-flat.",
        "A setting powder went over {ent}, leaving nothing that caught light.",
    ],
    "GLOSS": [
        "She slicked a clear balm over {ent} until it glistened.",
        "A wet-look topcoat went across {ent}.",
        "She added shine to {ent}, and it turned glassy.",
        "{Ent} was glossed until it looked freshly damp.",
        "She patted a dewy serum onto {ent}.",
        "A layer of gloss made {ent} catch every bit of light.",
        "She brushed something reflective over {ent}.",
        "{Ent} was left looking wet and lit from within.",
        "She topped {ent} with a liquid-looking sheen.",
        "A clear shine went over {ent}, and it began to gleam.",
    ],
    "SOFTEN": [
        "She buffed {ent} down to a soft, silky lustre.",
        "The shine on {ent} was knocked back to a gentle glow.",
        "She pressed {ent} into a half-shine, half-blur surface.",
        "{Ent} settled somewhere between flat and wet.",
        "She smoothed {ent} until it held a quiet sheen.",
        "A light buffing gave {ent} an understated silk-like surface.",
        "She took {ent} to a muted, low-key lustre.",
        "{Ent} ended up neither dry nor glossy, just softly reflective.",
        "She worked {ent} into a smooth semi-gloss.",
        "The surface of {ent} was softened to a subtle, powdery shine.",
    ],
    "BUILD": [
        "She went back over {ent}, layering until nothing showed through.",
        "A second and third pass took {ent} to full opacity.",
        "She built {ent} up until the colour was solid.",
        "{Ent} was layered until it read completely covered.",
        "She kept packing product onto {ent} until it was dense.",
        "More was added to {ent}, and the coverage went total.",
        "She doubled the layer on {ent}.",
        "{Ent} was taken all the way up to full strength.",
        "She reinforced {ent} until nothing underneath was visible.",
        "Another heavy pass made {ent} fully opaque.",
    ],
    "SHEER": [
        "She buffed most of it off {ent}, leaving only a wash.",
        "{Ent} was diffused down until the skin showed through again.",
        "She blended {ent} out until barely anything remained.",
        "Most of the layer on {ent} was taken back off.",
        "She sheered {ent} down to a translucent trace.",
        "A clean brush thinned {ent} to a whisper.",
        "{Ent} was softened until it was only faintly there.",
        "She wiped {ent} back to a see-through film.",
        "The density on {ent} was reduced to almost nothing.",
        "She feathered {ent} out until it was nearly transparent.",
    ],
    "REMOVE": [
        "She took everything off {ent} with a single wipe.",
        "{Ent} was cleaned back to nothing at all.",
        "A cotton pad stripped {ent} completely.",
        "She removed every trace of product from {ent}.",
        "{Ent} was returned to bare skin.",
        "She wiped {ent} clean and started over.",
        "Everything that had been on {ent} came off.",
        "{Ent} was left completely undone.",
        "She erased what was on {ent} entirely.",
        "A remover cloth took {ent} back to blank.",
    ],
    "NOOP": [
        "She stepped back and checked the mirror.",
        "The light in the room shifted a little.",
        "She rinsed the brush and set it down.",
        "There was a pause while she considered.",
        "She tucked a strand of hair behind her ear.",
        "The radiator clicked on somewhere behind her.",
        "She tilted her head and looked again.",
        "Her phone buzzed on the counter, ignored.",
        "She wiped her hands on a towel.",
        "A moment passed with nothing done.",
    ],
    # Mentions a colour (and sometimes a region) without changing any state.
    # This is the anti-keyword control: bag-of-words probes are misled here.
    "DISTRACTOR": [
        "She picked up {mcol} and, after a moment, put it back unopened.",
        "{Mcol} sat in the tray, still untouched.",
        "She thought about {mcol} for {ment}, then decided against it.",
        "A tube of {mcol} rolled to the edge of the counter, unused.",
        "She held {mcol} up to the light and shook her head.",
        "{Mcol} was ruled out before the brush ever touched it.",
        "She considered {mcol} on {ment} and dismissed the idea.",
        "The palette beside her held {mcol}, which she never reached for.",
        "She almost went for {mcol}, then decided not to.",
        "{Mcol} stayed capped for the whole routine.",
    ],
}

_SLOT = re.compile(r"\{([A-Za-z]+)\}")


def fill(template: str, values: dict[str, str]) -> str:
    """Fill {slot} / {Slot} (capitalised) placeholders."""

    def sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key[0].isupper():
            v = values[key[0].lower() + key[1:]]
            return v[0].upper() + v[1:]
        return values[key]

    return _SLOT.sub(sub, template)


def sanity_check() -> None:
    for bank in (ENTITY_PHRASES, COLOR_PHRASES, FINISH_PHRASES,
                 COVERAGE_PHRASES, TEMPLATES):
        for key, variants in bank.items():
            assert len(variants) == 10, f"{key} has {len(variants)} entries"
            assert len(set(variants)) == 10, f"{key} has duplicates"


sanity_check()
