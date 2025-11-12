# Pipeline Description


<!-- The following is a "mermaid chart". It is automatically rendered by gitlab and github. Locally it can be previewed e.g. in vs code by this plugin: https://docs.mermaidchart.com/plugins/visual-studio-code.  -->

```mermaid
  graph TD
    A["Lot Preparation
    *(chimcla split-into-lots)*"] --> |separated lots|B[Form Image Preprocessing]
    B --> C[Bar-Separation]
    C --> D["Brightness evaluation
    (S-Value calculation)"]
    D -->F["Step History
            Evaluation
            *(chimcla_step
            _history_eval)*"]
    D -->G["Generation of
            Experimental Images
            *(chimcla_ced)*"]
    D --> E["Generation of
            CNN Training Data
            *(chimcla_create
            _work_images)*"]
```
