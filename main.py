import argparse
from env import LiarDiceEnv
from agents.basic_agent import BasicRuleAgent


def main(args):
    
    env = LiarDiceEnv(num_players=args.num_players, render_mode="human" if args.render else None)
    agents = {agent: BasicRuleAgent(agent, args.num_players) for agent in env.possible_agents}
    
    NUM_MATCHES = 2
    for match in range(NUM_MATCHES):
        print(f"\n\n--- MATCH {match + 1} STARTING ---")
        env.reset()
        
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                # Agent is done for this round, step with a dummy action
                action = None
            else:
                agent_obj = agents[agent_id]
                action = agent_obj.get_action(observation)
                print(f"Agent {agent_id} acts: {action}")
            
            env.step(action)
        
        print(f"--- MATCH {match + 1} ENDED ---")
        print(f"Final Penalties: {env.penalties}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--num_players", type=int, default=4, help="Number of players")
    args = parser.parse_args()
    
    main(args)