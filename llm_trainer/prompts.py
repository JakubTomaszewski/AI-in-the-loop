STRATEGY_GENRATION_PROMPT = """You are provided with a summary of prompts used for generating synthetic images along with the performance of the image classification model trained using those synthetic images for a set of different classes.
The model is trained using contrastive learning, so the quality of the synthetic images is crucial for the model's performance.

<class_information>
{class_information}
</class_information>

<instructions>
Analyze the prompts and the performance of the model and come up with a strategy for generating new prompts for each class that will potentially improve the model's performance.
Those strategies should be a set of guidelines for generating prompts that will be used for generating synthetic images. Those synthetic images are then used for training an image classification model.

Rules:
1. You MUST analyze the performance of the model and the summary of the prompts to generate a strategy for each class.
2. You MUST generate a strategy for each class based on the performance of the model and the summary of the prompts.
3. Be creative and think about how the prompts can be improved to generate more relevant synthetic images for each class. 
4. The strategies have to be relatively short and concise but detailed and specific to each class.
5. DO NOT include information about the current prompts or the performance of the model in the strategy. Only generate a strategy for generating new prompts.

Do not mention the sks token in the strategy. It will be added automatically when generating prompts.
</instructions>

<example>
Input:
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

Output:
{{
    "class1": "Generate prompts depicting the object in an urban and sunny environment. For example, 'A high resolution image of a sks {{object_category}}' + details about the urban environment.",
    "class2": "Add different viewing angles and perspectives to the prompts. For example, 'A high resolution image of a sks {{object_category}}' + details about the object orientation or perspective.",
    "class3": "Generate prompts that have different, forest-like backgrounds. For example, 'A sks {{object_category}}' + details about the forest-like background."
}}
</example>

<output_format>
{output_format}
</output_format>
"""


PROMPT_GENERATION_PROMPT = """You are provided with a strategy for generating prompts. The strategy is a set of guidelines for generating prompts that will be used for generating synthetic images. Those synthetic images are then used for training an image classification model using contrastive learning.

<prompt_strategy>
{prompt_strategy}
</prompt_strategy>

<instructions>
Based on the provided strategy, generate new prompts for each class that will be used for generating synthetic images.
Each of the prompts MUST include the sks token and the object category name which is: {object_category}. The rest of the prompt should be based on the strategy provided.
The starting part of the prompt should be: "A sks {{object_category}}" + details about the object or scene.

Rules to follow:

1. Generate prompts that are ONLY relevant to the strategy provided.
2. You MUST generate exactly {num_prompts} prompts for each class based on the strategy provided!
3. You MUST enumerate the prompts to keep track of them. I.e. "Prompt 1: ...", "Prompt 2: ...", etc.
4. Each prompt can have a maximum of 15 words.
5. The prompts MUST be short and concise.
6. Each prompt MUST be relevant to the base class. For instance, if the class is "mug with dots", the prompts should be related to the base class being a "mug", not the "dots".
7. ONLY modify the background, scene, or setting around the object. DO NOT change the object itself. I.e. don't change the color, material or anything else related to the object.

</instructions>

<example>
Examples:

## Example 1
Strategy: "Generate prompts that have different, forest-like backgrounds. For example, 'A ... sks {{object_category}}' + details about the forest-like background."
Your output: [
    "Prompt 1: A a sks cup in a forest, from the top view",
    "Prompt 2: A bright image of a sks cup in at a lake, surrounded by trees and grass",
]

## Example 2
Strategy: "Generate prompts that are more detailed and specific to the class. For example, 'A high resolution image of a sks {{object_category}}' + details about the object or scene. Add different object orientations to the prompts. For example, 'A bright image of a sks {{object_category}}' + details about the object orientation or perspective." 
Your output: [
    "Prompt 1: A sks backpack in a school from the side, with a shadow on the ground"
    "Prompt 2: A high resolution image of a sks backpack on a desk, from the top view, with a bright light shining on it"
]
</example>

<output_format>
{output_format}

Each prompt should be a separate string.

Example:
[
    "Prompt 1: A high resolution image of a sks cup in a forest, from the top view",
    "Prompt 2: A sks cup in a forest, from the top view, surrounded by trees and multiple animals",
    ...
]

</output_format>
"""


PROMPT_SUMMARIZATION_PROMPT = """You are provided with a list of prompts used for generating synthetic images. Those synthetic are then used for training an image classification model using contrastive learning.
Each prompt is a description of the image that should be generated in the format of: "A ... sks {{object_category}}" + details about the object or scene.
For example:
- "A sks mug on a table"
- "A high-resolution and overexposed image of a sks ball in a grassy field"

Note, "sks" is a particular token which you should ignore.

<prompts>
{prompts}
</prompts>

<instructions>
Analyze the prompts and summarize them by identifying various aspects of the prompts that are important for generating synthetic images. Detect patterns, commonalities, and differences between the prompts.

Rules:
1. Identify the common aspects and details of the prompts that are important for generating synthetic images.
2. Describe the backgrounds, scenes, and scenarios in the prompts.
3. You MUST be concise and use a maximum of 40 words.
4. DO NOT include any irrelevant information.
5. DO NOT mention the sks token in the summary.
6. Do not focus on the object itself, but rather on the background, scene, and scenarios.

</instructions>

<example>
Input:
[
    'A high resolution image of a sks {{object_category}} laying in a garden, surrounded by flowers. Top view.', 
    'A bright image of a sks {{object_category}} in a forest, from the top view, surrounded by trees and multiple animals', 
    'A sks {{object_category}}' with a dark background, from the upper view', 
    'A picture containing a sks {{object_category}} on a grassy field'
]

Output:
'The prompts contain backgrounds mainly connected with nature, including trees, forests, gardens, and grassy fields. The objects are often described as being in a top view, with some prompts mentioning multiple animals. Yet, most objects are associated with nature and plants. The prompts mainly include bright images.'
</example>

<output_format>
Only write the summary of the prompts. Do not write any code.
</output_format>
"""
