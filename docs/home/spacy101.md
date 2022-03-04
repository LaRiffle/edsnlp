# SpaCy 101

EDS-NLP is a SpaCy library. To use it, you will need to familiarise yourself with some key SpaCy concepts.

!!! tip "Skip if you're familiar with SpaCy"

    This page is intended as a crash course for the very basic SpaCy concepts that are needed to use EDS-NLP.
    If you've already used SpaCy, you should probably skip to the next page.

In a nutshell, SpaCy offers three things:

- a convenient abstraction with a language-dependant, rule-based, deterministic and non-destructive tokenizer
- a rich set of rule-based and trainable components
- a configuration and training system

[![SpaCy](https://raw.githubusercontent.com/explosion/spaCy/master/website/src/images/logo.svg){ align=right width="30%" }](https://spacy.io/usage/spacy-101)

We will focus on the first item.

Be sure to checkout [SpaCy's crash course page](https://spacy.io/usage/spacy-101) for more information on the possibilities offered by the library.

## Resources

The [SpaCy documentation](https://spacy.io/) is one of the great strengths of the library.
In particular, you should check out the ["Advanced NLP with SpaCy" course](https://course.spacy.io/en/),
which provides a more in-depth presentation.

## SpaCy in action

Consider the following minimal example:

```python
import spacy  # (1)

# Initialise a SpaCy pipeline
nlp = spacy.blank("fr")  # (2)

text = "Michel est un penseur latéral."  # (3)

# Apply the pipeline
doc = nlp(text)  # (4)

doc.text
# Out: 'Michel est un penseur latéral.'
```

1.  Import SpaCy...
2.  Load a pipeline. In SpaCy, the `nlp` object handles the entire processing.
3.  Define a text you want to process.
4.  Apply the pipeline and get a SpaCy [`Doc`](https://spacy.io/api/doc) object.

We just created a SpaCy pipeline and applied it to a sample text. It's that simple.

Note that we use SpaCy's "blank" NLP pipeline here.
It actually carries a lot of information,
and defines SpaCy's language-dependent, rule-based tokenizer.
However,

!!! note "Non-destructive processing"

    In EDS-NLP, just like SpaCy, non-destructiveness is a core principle.
    Your detected entities will **always** be linked to the **original text**.

    In other words, `#!python nlp(text).text == text` is always true.

The first two lines import SpaCy and load a "blank" French-language NLP object.

### The `Doc` abstraction

The `doc` object carries the result of the entire processing.
It's the most important abstraction in SpaCy,
and holds a token-based representation of the text along with the results of every pipeline components.
It also keeps track of the input text in a non-destructive manner, meaning that
`#!python doc.text == text` is always true.

```python
# ↑ Omitted code above ↑

# Text processing in SpaCy is non-destructive
doc.text == text  # (1)

# You can access a specific token
token = doc[2]  # (2)

# And create a Span using slices
span = doc[:3]  # (3)

# Entities are tracked in the ents attribute
doc.ents  # (4)
# Out: (,)
```

1.  This feature is a core principle in SpaCy. It will always be true in EDS-NLP.
2.  `token` is a [`Token`](https://spacy.io/api/token) object referencing the third token
3.  `span` is a [`Span`](https://spacy.io/api/span) object referencing the first three tokens.
4.  We have not declared any entity recognizer in our pipeline, hence this attribute is empty.

### Adding pipeline components

You can add pipeline components with the `#!python nlp.add_pipe` method. Let's add two simple components to our pipeline.

```python hl_lines="5-6"
import spacy

nlp = spacy.blank("fr")

nlp.add_pipe("eds.sentences")  # (1)
nlp.add_pipe("eds.dates")  # (2)

text = "Le 2 février, Michel ne comprend pas l'inférence directe."

doc = nlp(text)
```

1. Like the name suggests, this pipeline is declared by EDS-NLP.
   `eds.sentences` is a rule-based sentence boundary prediction.
   See [its documentation](../pipelines/core/sentences.md) for detail.
2. Like the name suggests, this pipeline is declared by EDS-NLP.
   `eds.dates` is a date extraction and normalisation component.
   See [its documentation](../pipelines/misc/dates.md) for detail.

The `doc` object just became more interesting!

```python
# ↑ Omitted code above ↑

# We can split the document into sentences
doc.sents  # (1)
# Out: [Le 2 février, Michel ne comprend pas l'inférence directe.]

# And look for dates
doc.spans["dates"]  # (2)
# Out: [2 février]

span = doc.spans["dates"][0]  # (3)
span._.date  # (3)
# Out: "????-02-02"
```

1. In this example, there is only one sentence...
2. The `eds.dates` adds a key to the `doc.spans` attribute
3. `span` is a SpaCy `Span` object.
4. In SpaCy, you can declare custom extensions that live in the `_` attribute.
   Here, the `eds.dates` pipeline uses a `Span._.date` extension to persist the normalised date.

## Conclusion

This page is just a glimpse of a few possibilities offered by SpaCy. To get a sense of what SpaCy can help you achieve,
we **strongly recommend** you visit their [documentation](https://spacy.io/)
and take the time to follow the [SpaCy course](https://course.spacy.io/en/).

Be sure to check out [SpaCy's own crash course](https://spacy.io/usage/spacy-101){target="\_blank"}, which is an excellent read. It goes into more detail on what's possible with the library.