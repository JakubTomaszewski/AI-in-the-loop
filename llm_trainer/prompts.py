STRATEGY_GENRATION_PROMPT = """You are provided with a summary of prompts used for generating synthetic images along with the performance of the image classification model trained using those synthetic images for a set of different classes. This is denoted as the <class_information> section. The performance is a number between 0 and 1, where 1 is the best performance. The summary of the prompts is a concise description of the prompts used for generating synthetic images.

# Instructions
Analyze the prompts and the performance of the model and come up with a strategy for generating new prompts for each class that will potentially improve the model's performance.
Those strategies should be a set of guidelines for generating prompts that will be used for generating synthetic images. Those synthetic images are then used for training an image classification model.

## Rules:
1. You **must** analyze the performance of the model and the summary of the prompts to generate a strategy for each class. This information is provided in the <class_information> section.
2. You **must** generate a strategy for each class based on the performance of the model and the summary of the prompts.
3. Be creative and think about how the prompts can be improved to generate more relevant synthetic images for each class. 
4. The strategies **must** be short and concise but detailed and specific to each class.
5. **do not** include information about the current prompts or the performance of the model in the strategy. Only generate a strategy for generating new prompts.
6. You **must** **only** modify the background or setting around the object **do not** change the object itself. I.e. don't change the color, material or anything else related to the object.
7. Be specific about the background or setting around the object. **do not** generate general or vague strategies like "Emphasize reflections on glossy surfaces and incorporate subtle decorative elements to highlight design details."
8. The strategies **must** only guide the generation towards specific backgrounds and not towards specific features of the object itself. For instance, strategies mentioning "detailed", "glossy", "decorative", etc. elements of the object are not allowed.
9. Do not mention the sks token in the strategy. It will be added automatically when generating prompts.


# Output format
Format the output according to the following format:
{output_format}


# Example
<input_information>
"class_details": {{
    "class1": {{
    "performance": 0.5,
    "prompt_summary": "A summary of the prompts for class1"
    }},
    "class2": {{
    "performance": 0.8,
    "prompt_summary": "A summary of the prompts for class2"
    }},
    "class3": {{
    "performance": 0.3,
    "prompt_summary": "A summary of the prompts for class3"
    }}
}}
</input_information>

<output>
{{
    "class1": "Generate prompts depicting the object in an urban and sunny environment. For example, 'A high resolution image of a sks {{object_category}}' + details about the urban environment.",
    "class2": "Generate prompts that have different, forest-like backgrounds. For example, 'A sks {{object_category}}' + details about the forest-like background.",
}}
</output>
"""

STRATEGY_GENRATION_USER_PROMPT = """
<class_information>
{class_information}
</class_information>
"""


PROMPT_GENERATION_PROMPT = """You are provided with a strategy for generating prompts marked <prompt_strategy>. The strategy is a set of guidelines for generating prompts that will be used for generating synthetic images. Those synthetic images are then used for training an image classification model using contrastive learning.

# Instructions
Based on the provided strategy, generate new prompts for each class that will be used for generating synthetic images.
Each of the prompts **must** include the sks token and the object category name which is: "{object_category}". The rest of the prompt should be based on the strategy provided.
The starting part of the prompt should be: "A sks {{object_category}}" + details about the object or scene.

## Rules:
1. Generate prompts that are **only** relevant to the strategy provided in the <prompt_strategy> section.
2. You **must** generate exactly {num_prompts} prompts for each class based on the strategy provided!
3. You **must** enumerate the prompts to keep track of them. I.e. "Prompt 1: ...", "Prompt 2: ...", etc.
4. Each prompt can have a maximum of 15 words.
5. The prompts **must** be short and concise.
6. Each prompt **must** be relevant to the base class. For instance, if the class is "mug with dots", the prompts should be related to the base class being a "mug", not the "dots".
7. **only** modify the background or setting around the object. **do not** change the object itself. I.e. don't change the color, material or anything else related to the object.
8. The prompts **must** only guide the generation towards specific backgrounds and not towards specific features of the object itself. For instance, prompts mentioning "detailed", "glossy", "decorative", etc. elements of the object are not allowed.


# Output format
Each prompt should be a separate string.
{output_format}


# Examples

## Example 1
<prompt_strategy>
"Generate prompts that have different, forest-like backgrounds. For example, 'A ... sks {{object_category}}' + details about the forest-like background."
</prompt_strategy>

<output>
[
    "Prompt 1: A a sks cup in a forest, surrounded by trees and multiple animals",
    "Prompt 2: A bright image of a sks cup in at a lake, surrounded by trees and grass",
]
</output>


## Example 2
<prompt_strategy>
"Generate prompts depicting the object in a school environment. For example, 'A high resolution image of a sks {{object_category}}' + details about the school environment."
</prompt_strategy>

<output>
[
    "Prompt 1: A sks backpack in a school.",
    "Prompt 2: A high resolution image of a sks backpack on a desk in a classroom.",
    "Prompt 3: A sks backpack in a school, surrounded by students and teachers.",
]
</output>
"""

PROMPT_GENERATION_USER_PROMPT = """
<prompt_strategy>
{prompt_strategy}
</prompt_strategy>
"""


PROMPT_SUMMARIZATION_PROMPT = """You are provided with a list of prompts in the <prompts> section. Those prompts are used for generating synthetic images, which are then used for training an image classification model using contrastive learning.
Each prompt is a description of the image that should be generated in the format of: "A ... sks {{object_category}}" + details about the object or scene.

# Instructions
Analyze the prompts and summarize them by identifying various aspects of the prompts that are important for generating synthetic images. Detect patterns, commonalities, and differences between the prompts.

## Rules:
1. Analyze the prompts from the <prompts> section.
2. Identify the common aspects and details of the prompts that are important for generating synthetic images.
3. Describe the backgrounds, scenes, and scenarios in the prompts.
4. You **must** be concise and use a maximum of 40 words.
5. **do not** include any irrelevant information.
6. **do not** mention the sks token in the summary.
7. **do not** focus on the object itself, but rather on the background, scene, and scenarios.
8. you **must** ignore the "sks" token and focus on the rest of the prompt.


# Output format
Only write the summary of the prompts. Do not write any code.


# Example
<input_information>
[
    'A high resolution image of a sks {{object_category}} laying in a garden, surrounded by flowers.', 
    'A bright image of a sks {{object_category}} in a forest, surrounded by trees and multiple animals.', 
    'A sks {{object_category}}' with a dark background.', 
    'A picture containing a sks {{object_category}} on a grassy field.'
]
</input_information>

<output>
'The prompts contain backgrounds mainly connected with nature, including trees, forests, gardens, and grassy fields. The objects are often depicted in settings with multiple animals. Yet, most objects are associated with nature and plants.'
</output>
"""


PROMPT_SUMMARIZATION_USER_PROMPT = """
<prompts>
{prompts}
</prompts>
"""
