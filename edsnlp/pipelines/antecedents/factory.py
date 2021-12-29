from typing import Dict, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.antecedents import Antecedents, terms
from edsnlp.pipelines.terminations import termination

antecedents_default_config = dict(
    antecedents=terms.antecedents,
    termination=termination,
)


@Language.factory("antecedents", default_config=antecedents_default_config)
def create_component(
    nlp: Language,
    name: str,
    antecedents: List[str],
    termination: List[str],
    use_sections: bool = True,
    filter_matches: bool = False,
    attr: str = "LOWER",
    explain: str = False,
    on_ents_only: bool = True,
    regex: Optional[Dict[str, Union[List[str], str]]] = None,
    ignore_excluded: bool = False,
):
    return Antecedents(
        nlp,
        antecedents=antecedents,
        termination=termination,
        use_sections=use_sections,
        filter_matches=filter_matches,
        attr=attr,
        explain=explain,
        on_ents_only=on_ents_only,
        regex=regex,
        ignore_excluded=ignore_excluded,
    )
