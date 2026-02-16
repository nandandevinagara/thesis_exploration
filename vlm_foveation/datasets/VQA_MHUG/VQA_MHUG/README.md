# License
This dataset is licensed under an Attribution-NonCommercial-ShareAlike 4.0 International license ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).

# Attribution / Citation
If you use this dataset, please cite the corresponding publication:

Ekta Sood, Fabian Kögel, Florian Strohm, Prajit Dhar, and Andreas Bulling. 2021. VQA-MHUG: A Gaze Dataset to Study Multimodal Neural Attention in Visual Question Answering. In Proceedings of the 25th Conference on Computational Natural Language Learning, pages 27-43, Online. Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.conll-1.3

# VQA-MHUG Distribution

Contacts: Fabian Kögel (fabian.koegel@vis.uni-stuttgart.de), Ekta Sood (ekta.sood@vis.uni-stuttgart.de)

This dataset distribution contains four conditions of **M**ultimodal **HU**man **G**aze (**J**ointly **R**ecorded):
|             |     question-image pairs (3 recordings each)    |     stimuli source dataset |
|-------------|-------------------------------------------------|----------------------------|
| VQA-MHUG    | 3990                                            | VQAv2  (Goyal et al. 2017) |
| VQA-MHUG-JR | 800                                             | VQAv2 (Goyal et al. 2017)  |
| AiR-MHUG    | 195                                             | GQA (Hudson and Manning 2019)   |
| AiR-MHUG-JR | 150                                             | GQA (Hudson and Manning 2019)   |

VQA-MHUG is the main dataset, which contains human gaze on the **V**isual **Q**uestion **A**nswering v2 dataset (Goyal et al. 2017) as stimuli. We selected the stimuli by balancing difficulty for the machine (based on how two SOTA architectures performed on them), types of reasoning capability a question requires and overlap with other existing datasets of supplementary annotations. Participants were asked to answer the questions on the images. The questions and images were presented separately one after each other for unlimited viewing time while the eye movements of the participants were tracked using an EyeLink 1000 Plus eye tracker at 2kHz. For JR conditions questions and images were presented together on the same screen, so that saccadic regressions and interactions between the two modalities were captured. To our knowledge this dataset is the largest task-specific multimodal eye tracking resource on VQA and the only one that includes human attention on questions.

The AiR conditions are only for comparison to an existing attention dataset (AiR-D, Chen et al. 2020), that is the most similar in recording paradigm. AiR-D is currently the only other dataset on VQA that uses real eye tracking, albeit using a VR-glasses tracker.
AiR-D is based on stimuli from a different source dataset GQA (Hudson and Manning 2019) and is therefore kept separate from the VQA set.

### Contents

We provide 3 gaze data recordings of different participants on each sample in 4 formats:
- raw fixations of both eyes on the full stimulus
- attention maps on images (generation script)
- attention maps on text (generation script)
- scanpath on images (generation script)

Additionally we provide the answers of the participants and their VQAv2 accuracy as well as the tags we used to select our subset of the VQAv2: reasoning types and difficulty scores.

The directories `mhug` and `mhug-jr` contain pickled pandas dataframes for the raw fixations in stimuli coordinates, bounding boxes (image, question, individual word positions on the stimuli) and participant answers. 
The files 'difficulty_scores.pickle' and 'reasoning_types.pickle' contain all tags for all conditions.

All files use the official VQAv2 or GQA ids as first index level and sometimes participant id or fixation id as additional levels. They can be loaded with `mhug = pandas.read_pickle(FILE)` and queried with `mhug.loc[QID, PID, ...]`. Which levels are available can be checked py printing the data frame or `mhug.index`.

### Generation Script
To obtain the other 3  formats (image/text attention maps and scanpaths), the script `generate_deliverables.py` needs to be run in the command line. `--help` prints all options, you can choose one or more conditions and formats and the output path. Additionally there is a switch to scale attention maps by the fixation duration.

