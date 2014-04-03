# Intro
To get an idea about theoretical machines see the entries for [finite state
machines](https://en.wikipedia.org/wiki/Finite-state_machine) or [Turing
machines](https://en.wikipedia.org/wiki/Turing_machine) on Wikipedia.

This project tries to implement some of these machines.  A command line
interface to load and run machines is provided.  To start it run the file
`automata.py`.  To see a summary of the command line options run `autmoata.py
--help`.

# Input format
Not documented at the moment.  See the respective to_json methods to get an
idea.  Here is a small example (which can also be seen in
`machines/fsm-test.json`)

```json
{
  "alphabet": ["a", "b", "c"],
  "final_states": [10],
  "start_state": 0,
  "table": {
           "0": { "a": 1, "c": 3 },
           "1": { "b": 2 },
           "2": { "c": 10 },
           "3": { "b": 4 },
           "4": { "a": 10 }
           }
}
```

There is a plain text format for the machines, too.  It's also not documented
but there are examples in the `machines` directory.  The format is derived
from some initial ideas to represent turing machnies in text files.
