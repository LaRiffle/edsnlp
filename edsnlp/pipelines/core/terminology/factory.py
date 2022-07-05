from typing import Any, Dict, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.core.terminology import TerminologyMatcher

DEFAULT_CONFIG = dict(
    terms=None,
    regex=None,
    attr="TEXT",
    ignore_excluded=False,
    algorithm="exact",
    algorithm_config={},
)


@Language.factory(
    "eds.terminology",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    label: str,
    terms: Optional[Dict[str, Union[str, List[str]]]],
    attr: Union[str, Dict[str, str]],
    regex: Optional[Dict[str, Union[str, List[str]]]],
    ignore_excluded: bool,
    algorithm: str,
    algorithm_config: Dict[str, Any],
):
    assert not (terms is None and regex is None)

    if terms is None:
        terms = dict()
    if regex is None:
        regex = dict()

    return TerminologyMatcher(
        nlp,
        label=label,
        terms=terms,
        attr=attr,
        regex=regex,
        ignore_excluded=ignore_excluded,
        algorithm=algorithm,
        algorithm_config=algorithm_config,
    )