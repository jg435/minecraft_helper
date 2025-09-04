# minecraft_helper

## How this works

This work is based on the MCP minecraft done by yuniko software. I have built an orchestration system that takes advantage of this MCP server. We have 3 bots that can be initialised: a builder, a farmer and a collector (it collects resources that you ask it to collect). Based on the user's command, the supervisor determines which bot(s) to initialise and gives them tasks to complete. 

We connect to minecraft via LAN. Make sure that you have the right LAN port set up and that you use a minecraft version equal to and below 1.21.5

## Essential Changes

Feel free to add your own openrouter key to line 24
Make sure your minecraft lan port is set to 58503 
Make sure to change the bots default coordinates in line 219 from  (91, 64, 155) to whatever spot you want them to come to
Write your commands for the bot(s) in line 530

## Optional Changes

Feel free to add public and secret keys from your langfuse project to this repo if you want observability

## Source
https://github.com/yuniko-software/minecraft-mcp-server 

## Issues:
The bots use A LOT of call to the llms. Every small action (eg what block is it currently looking at) is a tool call so the LLM is constantly viewing the results of the tool call. 

## TODO
Make this better
use ART agent: https://art.openpipe.ai/integrations/langgraph-integration (thanks for the rec Nate!)
