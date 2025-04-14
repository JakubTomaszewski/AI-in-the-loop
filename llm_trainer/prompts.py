STRATEGY_GENRATION_PROMPT = """You are provided with a summary of prompts used for generating synthetic images along with the performance of the image classification model trained using those synthetic images for a set of different classes.
The model is trained using contrastive learning, so the quality of the synthetic images is crucial for the model's performance.

<class_information>
{class_information}
</class_information>

<instructions>
Analyze the prompts and the performance of the model and come up with a strategy for generating new prompts for each class that will potentially improve the model's performance.
Those strategies should be a set of guidelines for generating prompts that will be used for generating synthetic images. Those synthetic images are then used for training an image classification model.
The strategy should be based on the performance of the model on different classes and should aim to improve the model's performance.
Be creative and think about how the prompts can be improved to generate more relevant synthetic images for each class. 
Make the strategies detailed and specific to each class and consider the performance of the model on that class.

Do not mention the <new1> token in the strategy. It will be added automatically when generating prompts.
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
    "class1": "Generate prompts that are more detailed and specific to the class. For example, 'A high resolution image of a <new1> {{object_category}}' + details about the object or scene.",
    "class2": "Add different object orientations to the prompts. For example, 'A bright image of a <new1> {{object_category}}' + details about the object orientation or perspective.",
    "class3": "Generate prompts that have different, forest-like backgrounds. For example, 'A <new1> {{object_category}}' + details about the forest-like background."
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
Each of the prompts MUST include the <new1> token and the object category name which is: {object_category}. The rest of the prompt should be based on the strategy provided.
The starting part of the prompt should be: "A <new1> {{object_category}}" + details about the object or scene.

Rules to follow:

1. Generate exactly {num_prompts} prompts for each class based on the strategy provided!
2. Enumerate the prompts to keep track of them. I.e. "Prompt 1: ...", "Prompt 2: ...", etc.
3. Each prompt can have a maximum of 15 words.
4. Ensure that the prompts are concise and relevant to the class. 
5. If the class is an object, it should not be doing an action. Instead, come up with a relevant scene or setting for the object.
6. Only modify the background, scene, or setting around the object. Do not change the object itself. I.e. don't change the color, material or anything else related to the object.
7. Be creative and think about how the prompts can be improved to generate more relevant synthetic images for each class.

</instructions>

<example>
Examples:

Strategy: "Generate prompts that have different, forest-like backgrounds. For example, 'A ... <new1> {{object_category}}' + details about the forest-like background."
Your output: "A high resolution image of a <new1> cup in a forest, from the top view"

Strategy: "Generate prompts that are more detailed and specific to the class. For example, 'A high resolution image of a <new1> {{object_category}}' + details about the object or scene. Add different object orientations to the prompts. For example, 'A bright image of a <new1> {{object_category}}' + details about the object orientation or perspective." 
Your output: [
    "Prompt 1: A detailed image of a <new1> backpack from the side, with a shadow on the ground"
    "Prompt 2: A high resolution image of a <new1> backpack from the top view, with a bright light shining on it"
]

Strategy: "Generate prompts that include more objects in the scene, as well as more detailed descriptions of the objects."
Your output: [
    "Prompt 1: A close-up image of a <new1> mug on a table, with a cup of coffee next to it, and a book in the background",
    "Prompt 2: A detailed image of a <new1> mug on a table edge, with a spoon next to it, and a window in the background, with sunlight shining through"
]
</example>

<output_format>
{output_format}

Each prompt should be a separate string.

Example:
[
    "Prompt 1: A high resolution image of a <new1> cup in a forest, from the top view",
    "Prompt 2: A bright image of a <new1> cup in a forest, from the top view, surrounded by trees and multiple animals",
    ...
]

</output_format>
"""


PROMPT_SUMMARIZATION_PROMPT = """You are provided with a list of prompts used for generating synthetic images. Those synthetic are then used for training an image classification model using contrastive learning.
Each prompt is a description of the image that should be generated in the format of: "A ... <new1> {{object_category}}" + details about the object or scene.
For example:
- "A <new1> mug on a table"
- "A high-resolution and overexposed image of a <new1> ball in a grassy field"

Note, "<new1>" is a particular token which you should ignore.

<prompts>
{prompts}
</prompts>

<instructions>
Analyze the prompts and summarize them by identifying various aspects of the prompts that are important for generating synthetic images. Detect patterns, commonalities, and differences between the prompts.
Be concise, but provide many common details about the prompts. Do not describe the object itself, but focus on the background, scene, and scenarios.
For example, you could summarize the prompts by identifying the following aspects:
- background and surrounding
- texture
- object orientation
You can also identify any other aspects that you think are important for generating synthetic images. Do not focus too much on the example aspects.

Important rules:
- Make the summaries short, concise and informative. 
- Do not include any irrelevant information.
- Do not mention the <new1> token in the summary.
</instructions>

<example>
Input:
[
    'A high resolution image of a <new1> {{object_category}} laying in a garden, surrounded by flowers. Top view.', 
    'A bright image of a <new1> {{object_category}} in a forest, from the top view, surrounded by trees and multiple animals', 
    'A <new1> {{object_category}}' with a dark background, from the upper view', 
    'A picture containing a <new1> {{object_category}} on a grassy field'
]

Output:
'The prompts contain backgrounds mainly connected with nature, including trees, forests, gardens, and grassy fields. The objects are often described as being in a top view, with some prompts mentioning multiple animals. Yet, most objects are associated with nature and plants. The prompts mainly include bright images.'
</example>

<output_format>
Only write the summary of the prompts. Do not write any code.
</output_format>
"""
