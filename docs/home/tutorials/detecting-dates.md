# Detecting dates

We now know how to match a terminology and qualify detected entities, which covers most use cases for a typical medical NLP project.
In this tutorial, we'll see how to use EDS-NLP to detect and normalise date mentions using [`eds.dates`](../../pipelines/misc/dates.md).

This can have many applications, for dating medical events in particular.
The [`eds.consultation_dates`](../../pipelines/misc/consultation-dates.md) component, for instance,
combines the date detection capabilities with a few simple patterns to detect the date of the consultation, when mentioned in clinical reports.

## Dates in clinical notes

Consider the following example:

=== "French"

    ```
    Le patient est admis le 21 janvier pour une douleur dans le cou.
    Il se plaint d'une douleur chronique qui a débuté il y a trois ans.
    ```

=== "English"

    ```
    The patient is admitted on January 21st for a neck pain.
    He complains about chronique pain that started three years ago.
    ```

Clinical notes contain many different types of dates. To name a few examples:

| Type     | Description                         | Examples                                         |
| -------- | ----------------------------------- | ------------------------------------------------ |
| Absolute | Explicit date                       | `2022-03-03`                                     |
| Partial  | Date missing the day, month or year | `le 3 janvier/on January 3rd`, `en 2021/in 2021` |
| Relative | Relative dates                      | `hier/yesterday`, `le mois dernier/last month`   |

## Extracting dates

The followings snippet adds the `eds.date` component to the pipeline:

```python
import spacy

nlp = spacy.blank("fr")
nlp.add_pipe("eds.dates")  # (1)

text = (
    "Le patient est admis le 21 janvier pour une douleur dans le cou.\n"
    "Il se plaint d'une douleur chronique qui a débuté il y a trois ans."
)

# Detecting dates becomes trivial
doc = nlp(text)

# Likewise, accessing detected dates is hassle-free
dates = doc.spans["dates"]  # (2)
```

1. The date detection component is declared with `eds.dates`
2. Dates are saved in the `#!python doc.spans["dates"]` key

After this, accessing dates and there normalisation becomes trivial:

```python
# ↑ Omitted code above ↑

dates  # (1)
# Out: [21 janvier, il y a trois ans]
```

1. `dates` is a list of SpaCy `Span` objects.

We can review each date and get its normalisation:

| `date.text`        | `date._.date` |
| ------------------ | ------------- |
| `21 janvier`       | `????-01-21`  |
| `il y a trois ans` | `TD-1095`     |

## What next?

The `eds.dates` pipeline component's role is merely to detect and normalise dates.
It is the user's responsibility to use this valuable information to build.

For instance, you could use this pipeline to date medical entities. Let's do that.

### A medical event tagger

Our pipeline will detect entities and events separately,
and we will post-process the output `Doc` object to determine
whether a given entity can be linked to a date.

```python
import spacy

nlp = spacy.blank("fr")
nlp.add_pipe("eds.sentences")
nlp.add_pipe("eds.dates")

config = dict(
    regex=dict(admission=["admissions?", "admise?", "prise? en charge"]),
    attr="LOWER",
)
nlp.add_pipe("eds.matcher", config=config)

text = (
    "Le patient est admis le 12 avril pour une douleur "
    "survenue il y a trois jours. "
    "Il avait été pris en charge l'année dernière."
)

doc = nlp(text)
```

At this point, the document is ready to be post-processed: its `ents` and `#!python spans["dates"]` are populated:

```python
# ↑ Omitted code above ↑

docs.ents
# Out: (admis, pris en charge)

doc.spans["dates"]
# Out: [12 avril, il y a trois jours, l'année dernière]
```

As a first heuristic, let's consider that an entity can be linked to a date if the two are in the same
sentence. In the case where multiple dates are present, we'll select the closest one.

```python title="utils.py"
from spacy.tokens import Span
from typing import List, Optional


def candidate_dates(ent: Span) -> List[Span]:
    """Return every dates in the same sentence as the entity"""
    return [date for date in ent.doc.spans["dates"] if date.sent == ent.sent]


def get_event_date(ent: Span) -> Optional[Span]:
    """Link an entity to the closest date in the sentence, if any"""

    dates = candidate_dates(ent)  # (1)

    if not dates:
        return

    dates = sorted(
        dates,
        key=lambda d: min(abs(d.start - ent.end), abs(ent.start - d.end)),
    )

    return dates[0]  # (2)
```

1. Get all dates present in the same sentence.
2. Sort the dates, and keep the first item.

We can apply this simple function:

```python
import spacy
from utils import get_event_date

nlp = spacy.blank("fr")
nlp.add_pipe("eds.sentences")
nlp.add_pipe("eds.dates")

config = dict(
    regex=dict(admission=["admissions?", "admise?", "prise? en charge"]),
    attr="LOWER",
)
nlp.add_pipe("eds.matcher", config=config)

text = (
    "Le patient est admis le 12 avril pour une douleur "
    "survenue il y a trois jours. "
    "Il avait été pris en charge l'année dernière."
)

doc = nlp(text)

for ent in doc.ents:
    print(ent, get_event_date(ent))
```

Which will output:

| `ent`          | `get_event_date(ent)` | `get_event_date(ent)._.date` |
| -------------- | --------------------- | ---------------------------- |
| admis          | 12 avril              | `????-04-12`                 |
| pris en charge | l'année dernière      | `TD-365`                     |