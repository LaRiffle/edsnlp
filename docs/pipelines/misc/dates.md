# Dates

The `eds.dates` pipeline's role is to detect and normalise dates within a medical document.
We use simple regular expressions to extract date mentions.

## Scope

The `eds.dates` pipeline finds absolute (eg `23/08/2021`) and relative (eg `hier`, `la semaine dernière`) dates alike. It also handles mentions of duration.

| Type       | Example                       |
| ---------- | ----------------------------- |
| `absolute` | `3 mai`, `03/05/2020`         |
| `relative` | `hier`, `la semaine dernière` |
| `duration` | `pendant quatre jours`        |

See the [tutorial](../../tutorials/detecting-dates.md) for a presentation of a full pipeline featuring the `eds.dates` component.

## Usage

```python
import spacy

import pendulum

nlp = spacy.blank("fr")
nlp.add_pipe("eds.dates")

text = (
    "Le patient est admis le 23 août 2021 pour une douleur à l'estomac. "
    "Il lui était arrivé la même chose il y a un an pendant une semaine. "
    "Il a été diagnostiqué en mai 1995."
)

doc = nlp(text)

dates = doc.spans["dates"]
dates
# Out: [23 août 2021, il y a un an, mai 1995]

dates[0]._.date.to_datetime()
# Out: 2021-08-23T00:00:00+02:00

dates[1]._.date.to_datetime()
# Out: None

note_datetime = pendulum.datetime(2021, 8, 27, tz="Europe/Paris")

dates[1]._.date.to_datetime(note_datetime=note_datetime)
# Out: 2020-08-27T00:00:00+02:00

date_2_output = dates[2]._.date.to_datetime(
    note_datetime=note_datetime,
    infer_from_context=True,
    tz="Europe/Paris",
    default_day=15,
)
date_2_output
# Out: 1995-05-15T00:00:00+02:00

doc.spans["durations"]
# Out: [pendant une semaine]
```

## Declared extensions

The `eds.dates` pipeline declares one [spaCy extension](https://spacy.io/usage/processing-pipelines#custom-components-attributes) on the `Span` object: the `date` attribute contains a parsed version of the date.

## Configuration

The pipeline can be configured using the following parameters :

::: edsnlp.pipelines.misc.dates.factory.create_component
    options:
        only_parameters: true

## Authors and citation

The `eds.dates` pipeline was developed by AP-HP's Data Science team.
