import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union, Annotated
from enum import Enum
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from mcp_use import MCPAgent, MCPClient
from langfuse import observe, Langfuse

langfuse = Langfuse(
  secret_key="sk-lf-17fa378e-bd7b-4677-b845-e54740a8d842",
  public_key="pk-lf-6af19004-5bbf-4762-a6fd-cb012bcc25e7",
  host="https://us.cloud.langfuse.com"
)


class TaskType(Enum):
    """Available task types for supervisor selection"""
    BUILDER = "builder"
    RESOURCE_COLLECTOR = "resource_collector" 
    FARMER = "farmer"
    CUSTOM = "custom"

class MinecraftState(MessagesState):
    """State for Minecraft multi-agent system"""
    coordinates: tuple = (84, 64, 193)
    task_results: Dict[str, Any] = {}
    active_agents: List[str] = []

@dataclass
class TaskConfig:
    """Configuration for a specific task"""
    name: str
    username: str
    port: str
    max_steps: int = 100
    coordinates: Optional[tuple] = None
    parameters: Dict[str, Any] = None

@dataclass
class TaskResult:
    """Result of a task execution"""
    task_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0

class TaskFramework(ABC):
    """Abstract base class for all task frameworks"""
    
    def __init__(self, config: TaskConfig, llm):
        self.config = config
        self.llm = llm
        self.agent = None
    
    def create_bot_client(self) -> MCPAgent:
        """Create a bot client with framework-specific configuration"""
        mcp_config = {
            "mcpServers": {
                "minecraft": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "github:yuniko-software/minecraft-mcp-server",
                        "--host",
                        "localhost",
                        "--port",
                        str(self.config.port),
                        "--username",
                        self.config.username
                    ]
                }
            }
        }
        client = MCPClient.from_dict(mcp_config)
        return MCPAgent(llm=self.llm, client=client, max_steps=self.config.max_steps)
    
    @abstractmethod
    def generate_task_prompt(self) -> str:
        """Generate the task-specific prompt for the bot"""
        pass
    
    @observe
    async def execute(self) -> TaskResult:
        """Execute the task and return result"""
        import time
        start_time = time.time()
        
        try:
            print(f"🤖 Starting {self.config.name}...")
            self.agent = self.create_bot_client()
            prompt = self.generate_task_prompt()
            
            result = await self.agent.run(prompt)
            execution_time = time.time() - start_time
            
            print(f"✅ {self.config.name} completed in {execution_time:.2f}s")
            return TaskResult(
                task_name=self.config.name,
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ {self.config.name} failed after {execution_time:.2f}s: {e}")
            return TaskResult(
                task_name=self.config.name,
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time
            )

class BuilderFramework(TaskFramework):
    """Framework for construction and building tasks"""
    
    @observe
    def generate_task_prompt(self) -> str:
        coords = self.config.coordinates or (84, 64, 193)
        params = self.config.parameters or {}
        
        structure_type = params.get('structure_type', 'mud house')
        material = params.get('material', 'dirt blocks')
        width = params.get('width', 5)
        depth = params.get('depth', 5)
        height = params.get('height', 3)
        features = params.get('features', ['door opening', 'windows'])
        
        return f"""
        Build a {structure_type} using {material} around {coords[0]}, {coords[1]}, {coords[2]}. The structure should be:
        1. {width}x{depth} blocks wide and deep
        2. {height} blocks tall
        3. Include: {', '.join(features)}
        4. Start building at coordinates around {coords[0]}, {coords[1]}, {coords[2]}
        
        Use {material} for construction and follow architectural best practices.
        """

class ResourceCollectionFramework(TaskFramework):
    """Framework for resource gathering and collection tasks"""
    
    @observe
    def generate_task_prompt(self) -> str:
        coords = self.config.coordinates or (84, 64, 193)
        params = self.config.parameters or {}
        
        resource_type = params.get('resource_type', 'wood')
        target_amount = params.get('target_amount', 5)
        storage_method = params.get('storage_method', 'chest')
        search_radius = params.get('search_radius', 50)
        
        return f"""
        Collect {resource_type} around {coords[0]}, {coords[1]}, {coords[2]}. Your tasks:
        1. Find the nearest {resource_type} sources within {search_radius} blocks
        2. Extract/mine/collect the {resource_type}
        3. Try to collect at least {target_amount} {resource_type} blocks/items
        4. Store collected resources in a {storage_method}
        5. Report collection results via chat
        6. Start search at coordinates around {coords[0]}, {coords[1]}, {coords[2]}
        
        Focus on efficient collection and proper resource management.
        """

class FarmingFramework(TaskFramework):
    """Framework for agricultural and farming tasks"""
    
    @observe
    def generate_task_prompt(self) -> str:
        coords = self.config.coordinates or (84, 64, 193)
        params = self.config.parameters or {}
        
        farm_size = params.get('farm_size', '5x5')
        crop_type = params.get('crop_type', 'any available seeds')
        irrigation = params.get('irrigation', True)
        preparation_only = params.get('preparation_only', False)
        
        irrigation_text = "Look for water nearby or create a water source" if irrigation else "No irrigation needed"
        planting_text = "prepare the farmland for future planting" if preparation_only else "plant seeds if available"
        
        return f"""
        Start a {farm_size} farm around {coords[0]}, {coords[1]}, {coords[2]} for {crop_type}. Your tasks:
        1. Find a flat area of at least {farm_size} blocks
        2. Clear the area (remove grass, stones, obstacles)
        3. Create farmland by tilling or placing dirt blocks
        4. {irrigation_text}
        5. Plant {crop_type} in organized rows if available
        6. If no seeds available, {planting_text}
        7. Position farm close to coordinates {coords[0]}, {coords[1]}, {coords[2]}
        
        Focus on creating sustainable, organized farmland with proper spacing.
        """

class MinecraftOrchestrationService:
    """Service for orchestrating multiple Minecraft bot frameworks with supervisor control
    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"""
    
    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(
            model="anthropic/claude-sonnet-4",
            base_url="https://openrouter.ai/api/v1",
            api_key="ADD YOUR OWN KEY HERE",
        )
        self.frameworks: Dict[str, TaskFramework] = {}
        self.task_results: List[TaskResult] = []
        self.default_coordinates = (91, 64, 155)
        self.default_port = "58503"
    
    def register_framework(self, framework_id: str, framework: TaskFramework):
        """Register a task framework with the orchestration service"""
        self.frameworks[framework_id] = framework
    
    def create_builder_framework(self, config: TaskConfig) -> BuilderFramework:
        """Create and configure a builder framework"""
        return BuilderFramework(config, self.llm)
    
    def create_resource_collector_framework(self, config: TaskConfig) -> ResourceCollectionFramework:
        """Create and configure a resource collection framework"""
        return ResourceCollectionFramework(config, self.llm)
    
    def create_farming_framework(self, config: TaskConfig) -> FarmingFramework:
        """Create and configure a farming framework"""
        return FarmingFramework(config, self.llm)
    
    @observe
    async def execute_framework(self, framework_id: str) -> TaskResult:
        """Execute a specific framework by ID"""
        if framework_id not in self.frameworks:
            raise ValueError(f"Framework {framework_id} not registered")
        
        framework = self.frameworks[framework_id]
        result = await framework.execute()
        self.task_results.append(result)
        return result
    
    @observe
    async def execute_all_frameworks(self, concurrent: bool = True) -> List[TaskResult]:
        """Execute all registered frameworks"""
        if concurrent:
            tasks = [self.execute_framework(fid) for fid in self.frameworks.keys()]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to TaskResult objects
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    framework_id = list(self.frameworks.keys())[i]
                    processed_results.append(TaskResult(
                        task_name=framework_id,
                        success=False,
                        result=None,
                        error=str(result)
                    ))
                else:
                    processed_results.append(result)
            return processed_results
        else:
            results = []
            for framework_id in self.frameworks.keys():
                result = await self.execute_framework(framework_id)
                results.append(result)
            return results
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of all task executions"""
        total_tasks = len(self.task_results)
        successful_tasks = sum(1 for r in self.task_results if r.success)
        total_time = sum(r.execution_time for r in self.task_results)
        
        return {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'failed_tasks': total_tasks - successful_tasks,
            'success_rate': (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            'total_execution_time': total_time,
            'average_execution_time': total_time / total_tasks if total_tasks > 0 else 0
        }

class MinecraftSupervisorService:
    """LangGraph-based supervisor for parallel Minecraft agent orchestration"""
    
    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(
            model="anthropic/claude-sonnet-4",
            base_url="https://openrouter.ai/api/v1",
            api_key="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",
        )
        self.orchestration_service = MinecraftOrchestrationService()
        self.graph = self._create_supervisor_graph()
    
    def _create_handoff_tools(self):
        """Create handoff tools for agent delegation"""
        
        @tool
        def assign_to_builder(task_description: str) -> str:
            """Assign construction/building tasks to the builder agent"""
            return f"BUILDER:{task_description}"
        
        @tool  
        def assign_to_collector(task_description: str) -> str:
            """Assign resource collection/mining tasks to the collector agent"""
            return f"COLLECTOR:{task_description}"
            
        @tool
        def assign_to_farmer(task_description: str) -> str:
            """Assign farming/agriculture tasks to the farmer agent"""  
            return f"FARMER:{task_description}"
        
        return [assign_to_builder, assign_to_collector, assign_to_farmer]
    
    def _create_supervisor_graph(self):
        """Create the LangGraph supervisor workflow with natural parallel execution"""
        
        # Create handoff tools
        tools = self._create_handoff_tools()
        
        # Create supervisor agent
        supervisor_prompt = """You are a Minecraft bot supervisor. Analyze user requests and delegate tasks to appropriate agents by calling the assignment tools.

        Available agents:
        - assign_to_builder: For construction, building structures, houses, walls, etc.
        - assign_to_collector: For gathering resources like wood, stone, mining, etc.  
        - assign_to_farmer: For farming, agriculture, planting crops, preparing farmland, etc.
        
        You can call multiple assignment tools if the user wants multiple tasks done in parallel.
        Be specific about what each agent should do."""
        
        supervisor_agent = create_react_agent(
            self.llm,
            tools=tools,
            prompt=supervisor_prompt
        )
        
        # Create individual agent execution nodes
        @observe
        async def builder_agent(state: MinecraftState):
            """Execute building tasks"""
            messages = state["messages"]
            
            # Find builder assignment in messages
            for msg in reversed(messages):
                if "BUILDER:" in msg.content:
                    task = msg.content.split("BUILDER:")[-1].strip()
                    
                    config = TaskConfig(
                        name="Builder Task",
                        username="BuilderBot",
                        port="58503", 
                        coordinates=state["coordinates"],
                        parameters={'structure_type': 'custom', 'material': 'dirt blocks'}
                    )
                    
                    framework = self.orchestration_service.create_builder_framework(config)
                    framework.generate_task_prompt = lambda: f"Build: {task}. Start at {state['coordinates']}"
                    
                    result = await framework.execute()
                    state["task_results"]["builder"] = result
                    
                    return {
                        **state,
                        "messages": state["messages"] + [HumanMessage(content=f"✅ Builder completed: {result.success}")]
                    }
            
            return state
        
        @observe
        async def collector_agent(state: MinecraftState):
            """Execute resource collection tasks"""
            messages = state["messages"]
            
            for msg in reversed(messages):
                if "COLLECTOR:" in msg.content:
                    task = msg.content.split("COLLECTOR:")[-1].strip()
                    
                    config = TaskConfig(
                        name="Collector Task",
                        username="CollectorBot",
                        port="58503",
                        coordinates=state["coordinates"], 
                        parameters={'resource_type': 'wood', 'target_amount': 10}
                    )
                    
                    framework = self.orchestration_service.create_resource_collector_framework(config)
                    framework.generate_task_prompt = lambda: f"Collect: {task}. Start at {state['coordinates']}"
                    
                    result = await framework.execute()
                    state["task_results"]["collector"] = result
                    
                    return {
                        **state,
                        "messages": state["messages"] + [HumanMessage(content=f"✅ Collector completed: {result.success}")]
                    }
            
            return state
            
        @observe
        async def farmer_agent(state: MinecraftState):
            """Execute farming tasks"""
            messages = state["messages"]
            
            for msg in reversed(messages):
                if "FARMER:" in msg.content:
                    task = msg.content.split("FARMER:")[-1].strip()
                    
                    config = TaskConfig(
                        name="Farmer Task", 
                        username="FarmerBot",
                        port="58503",
                        coordinates=state["coordinates"],
                        parameters={'farm_size': '5x5', 'crop_type': 'any available seeds'}
                    )
                    
                    framework = self.orchestration_service.create_farming_framework(config)
                    framework.generate_task_prompt = lambda: f"Farm: {task}. Start at {state['coordinates']}"
                    
                    result = await framework.execute()
                    state["task_results"]["farmer"] = result
                    
                    return {
                        **state, 
                        "messages": state["messages"] + [HumanMessage(content=f"✅ Farmer completed: {result.success}")]
                    }
            
            return state
        
        def should_run_builder(state: MinecraftState) -> bool:
            """Check if builder agent should run"""
            messages = state["messages"]
            return any("BUILDER:" in msg.content for msg in messages)
        
        def should_run_collector(state: MinecraftState) -> bool:
            """Check if collector agent should run"""
            messages = state["messages"]
            return any("COLLECTOR:" in msg.content for msg in messages)
            
        def should_run_farmer(state: MinecraftState) -> bool:
            """Check if farmer agent should run"""
            messages = state["messages"]
            return any("FARMER:" in msg.content for msg in messages)
        
        # Build the graph - LangGraph will naturally parallelize agents in same superstep
        graph = StateGraph(MinecraftState)
        
        # Add nodes
        graph.add_node("supervisor", supervisor_agent)
        graph.add_node("builder_agent", builder_agent)
        graph.add_node("collector_agent", collector_agent) 
        graph.add_node("farmer_agent", farmer_agent)
        
        # Router function that returns list of nodes to execute
        def route_to_agents(state: MinecraftState) -> List[str]:
            """Route to appropriate agents based on assignments"""
            agents_to_run = []
            
            if should_run_builder(state):
                agents_to_run.append("builder_agent")
            if should_run_collector(state):
                agents_to_run.append("collector_agent") 
            if should_run_farmer(state):
                agents_to_run.append("farmer_agent")
                
            # If no agents assigned, end the workflow
            return agents_to_run if agents_to_run else [END]
        
        # Add conditional edges - LangGraph will run multiple agents in parallel
        graph.add_conditional_edges(
            "supervisor",
            route_to_agents
        )
        
        # All agents flow to END
        graph.add_edge("builder_agent", END)
        graph.add_edge("collector_agent", END)
        graph.add_edge("farmer_agent", END)
        
        # Start with supervisor
        graph.add_edge(START, "supervisor")
        
        return graph.compile()
    
    @observe
    async def handle_user_request(self, user_request: str, coordinates: tuple = (91, 64, 155)) -> Dict[str, Any]:
        """Handle user request through the supervisor - agents run in parallel naturally"""
        
        initial_state = MinecraftState(
            messages=[HumanMessage(content=user_request)],
            coordinates=coordinates,
            task_results={},
            active_agents=[]
        )
        
        # Execute the supervisor graph - LangGraph handles parallel execution
        result = await self.graph.ainvoke(initial_state)
        
        return {
            "success": True,
            "task_results": result.get("task_results", {}),
            "messages": [msg.content for msg in result.get("messages", [])],
            "coordinates": result.get("coordinates", coordinates),
            "agents_executed": list(result.get("task_results", {}).keys())
        }

async def main():
    # Demo both the original orchestration service and the new supervisor
    print("🚀 Minecraft Agent Systems Demo")
    print("=" * 60)
    
    # Test the LangGraph Supervisor Service
    supervisor_service = MinecraftSupervisorService()
    
    print("\n🎮 Testing LangGraph Supervisor with Parallel Execution")
    print("-" * 60)
    
    # Test single task
    print("\n1️⃣ Task Test:")
    result1 = await supervisor_service.handle_user_request("Build a small house of dirt thats 7 x 7 x 7, collect 20 wood and make me a wheat farm with irrigation")
    print(f"Agents executed: {result1['agents_executed']}")
    print(f"Messages: {result1['messages'][-2:]}")  # Show last 2 messages
    
if __name__ == "__main__":
    asyncio.run(main())