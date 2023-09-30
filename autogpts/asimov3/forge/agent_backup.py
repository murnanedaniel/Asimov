import json
import pprint

from forge.sdk import (
    Agent,
    AgentDB,
    Step,
    StepRequestBody,
    Workspace,
    ForgeLogger,
    Task,
    TaskRequestBody,
    PromptEngine,
    chat_completion_request,
)

LOG = ForgeLogger(__name__)


class ForgeAgent(Agent):
    """
    The goal of the Forge is to take care of the boilerplate code so you can focus on
    agent design.

    There is a great paper surveying the agent landscape: https://arxiv.org/abs/2308.11432
    Which I would highly recommend reading as it will help you understand the possabilities.

    Here is a summary of the key components of an agent:

    Anatomy of an agent:
         - Profile
         - Memory
         - Planning
         - Action

    Profile:

    Agents typically perform a task by assuming specific roles. For example, a teacher,
    a coder, a planner etc. In using the profile in the llm prompt it has been shown to
    improve the quality of the output. https://arxiv.org/abs/2305.14688

    Additionally baed on the profile selected, the agent could be configured to use a
    different llm. The possabilities are endless and the profile can be selected selected
    dynamically based on the task at hand.

    Memory:

    Memory is critical for the agent to acculmulate experiences, self-evolve, and behave
    in a more consistent, reasonable, and effective manner. There are many approaches to
    memory. However, some thoughts: there is long term and short term or working memory.
    You may want different approaches for each. There has also been work exploring the
    idea of memory reflection, which is the ability to assess its memories and re-evaluate
    them. For example, condensting short term memories into long term memories.

    Planning:

    When humans face a complex task, they first break it down into simple subtasks and then
    solve each subtask one by one. The planning module empowers LLM-based agents with the ability
    to think and plan for solving complex tasks, which makes the agent more comprehensive,
    powerful, and reliable. The two key methods to consider are: Planning with feedback and planning
    without feedback.

    Action:

    Actions translate the agents decisions into specific outcomes. For example, if the agent
    decides to write a file, the action would be to write the file. There are many approaches you
    could implement actions.

    The Forge has a basic module for each of these areas. However, you are free to implement your own.
    This is just a starting point.
    """

    def __init__(self, database: AgentDB, workspace: Workspace):
        """
        The database is used to store tasks, steps and artifact metadata. The workspace is used to
        store artifacts. The workspace is a directory on the file system.

        Feel free to create subclasses of the database and workspace to implement your own storage
        """
        super().__init__(database, workspace)

        self.action_chat_history = []
        self.planning_chat_history = []

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to create
        a task.

        We are hooking into function to add a custom log message. Though you can do anything you
        want here.
        """
        task = await super().create_task(task_request)
        LOG.info(
            f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
        )

        # Load up the prompt engine
        LOG.info("Loading prompt engine...")
        prompt_engine = PromptEngine("gpt-3.5-turbo")

        # Start with a system prompt
        LOG.info("Loading system prompt...")
        system_prompt_actor = prompt_engine.load_prompt("system-format_actor", abilities=self.abilities.list_abilities_for_prompt())
        system_prompt_planner = prompt_engine.load_prompt("system-format_planner", abilities=self.abilities.list_abilities_for_prompt())

        # Then, load the task prompt with the designated parameters
        LOG.info("Loading task prompt...")
        task_prompt_planner = prompt_engine.load_prompt("task-intro_planner", task = task.input)
        
        # Create message list
        LOG.info("Creating message list...")
        messages_planner = [
            {"role": "system", "content": system_prompt_planner},
            {"role": "user", "content": task_prompt_planner}
        ]

        self.planning_chat_history = messages_planner

        # Generate the chat response
        for attempt in range(3):
            try:
                LOG.info("Generating chat response...")
                chat_response_planner = await chat_completion_request(messages=messages_planner, model="gpt-3.5-turbo")
                LOG.info(pprint.pformat(f"Chat response: {chat_response_planner}"))

                answer = json.loads(chat_response_planner["choices"][0]["message"]["content"])
                initial_plan = answer["thoughts"]["plan"]
                break
            except Exception as e:
                LOG.error(f"Attempt {attempt+1} failed with error: {e}")
                if attempt == 2:
                    raise

        # Load into actor messages
        LOG.info("Loading actor messages...")
        task_prompt_actor = prompt_engine.load_prompt("task-intro_actor", task = task.input, plan = initial_plan)
        messages_actor = [
            {"role": "system", "content": system_prompt_actor},
            {"role": "user", "content": task_prompt_actor}
        ]

        self.action_chat_history = messages_actor

        return task

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        """
        The structure here is:
        1. Get the task and step info
        2. Run the action, given the chat history, and a current running plan
        3. Update the chat history with the action response
        4. Update the chat history with a printout of the action (called in the ability itself)
        5. Load the planning user step prompt, with the output of the action in the prompt
        6. Run the planning chat completion, and update the current running plan
        """
        
        task = await self.db.get_task(task_id)

        step = await self.db.create_step(
            task_id=task_id, input=step_request, is_last=False
        )
        LOG.info(pprint.pformat(f"Step created: {step.step_id}, input: {step.input}"))

        for message in self.action_chat_history:
            print("--------------------------------------------------------------------------------------------")
            print(f"{message['role']}: {message['content']}")
            print("--------------------------------------------------------------------------------------------")

        for attempt in range(3):
            try:
                LOG.info("Generating chat completion request...")
                chat_completion_kwargs = {
                    "messages": self.action_chat_history,
                    "model": "gpt-3.5-turbo"
                    # "model": "gpt-4"
                }

                chat_response = await chat_completion_request(**chat_completion_kwargs)
                LOG.info(pprint.pformat(f"Chat response: {chat_response}"))
                answer = json.loads(chat_response["choices"][0]["message"]["content"])

                self.action_chat_history.append({"role": "assistant", "content": json.dumps(answer)})

                # Extract the ability from the answer
                LOG.info("Extracting ability from answer...")
                ability = answer["ability"]
                LOG.info(pprint.pformat(f"Ability: {ability}"))

                # Run the ability and get the output
                LOG.info("Running ability...")
                output = await self.abilities.run_ability(
                    task_id, ability["name"], **ability["args"]
                )
                if output:
                    self.chat_history.append({"role": "assistant", "content": f"Here is the output of the ability {ability['name']} applied to {ability['args']}: {output}"})
                break

            except json.JSONDecodeError as e:
                # Handle JSON decoding errors
                LOG.info("Error decoding chat response.")
                LOG.error(pprint.pformat(f"Unable to decode chat response: {chat_response}"))
                if attempt == 2:
                    raise
            except Exception as e:
                # Handle other exceptions
                LOG.info("Error generating chat response.")
                LOG.error(pprint.pformat(f"Unable to generate chat response: {e}"))
                if attempt == 2:
                    raise

        step.is_last = True
        LOG.info("Step is due to finish.")

        # Set the step output to the "speak" part of the answer
        LOG.info("Setting step output...")
        if "thoughts" in answer and "speak" in answer["thoughts"]:
            step.output = answer["thoughts"]["speak"]
        elif "thoughts" in answer:
            step.output = answer["thoughts"]
        else:
            step.output = answer

        user_message = prompt_engine.load_prompt("user-step_planner", step_output = answer)
        self.planning_chat_history.append({"role": "user", "content": user_message})

        for message in self.planning_chat_history:
            print("--------------------------------------------------------------------------------------------")
            print(f"{message['role']}: {message['content']}")
            print("--------------------------------------------------------------------------------------------")

        for attempt in range(3):
            try:
                LOG.info("Generating chat completion request...")
                chat_completion_kwargs = {
                    "messages": self.planning_chat_history,
                    "model": "gpt-3.5-turbo"
                    # "model": "gpt-4"
                }

                chat_response = await chat_completion_request(**chat_completion_kwargs)
                LOG.info(pprint.pformat(f"Chat response: {chat_response}"))
                answer = json.loads(chat_response["choices"][0]["message"]["content"])
                user_message = prompt_engine.load_prompt("user-step_actor", plan = answer["thoughts"]["plan"])
                break

            except json.JSONDecodeError as e:
                # Handle JSON decoding errors
                LOG.info("Error decoding chat response.")
                LOG.error(pprint.pformat(f"Unable to decode chat response: {chat_response}"))
                if attempt == 2:
                    raise
            except Exception as e:
                # Handle other exceptions
                LOG.info("Error generating chat response.")
                LOG.error(pprint.pformat(f"Unable to generate chat response: {e}"))
                if attempt == 2:
                    raise

        self.action_chat_history.append({"role": "user", "content": user_message})

        if ability["name"] == "finish":
            step.is_last = True
            LOG.info("Step is due to finish.")

        LOG.info("--------------------------------------------------------------------------------------------")

        return step
