import torch
from network import ActorCritic
from scene_loader import THORDiscreteEnvironment as Environment
import torch.nn.functional as F
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
scope = 'bathroom_02'

scene_scope = 'bathroom_02'
task_scopes = ['26']

'''
    ls_scene_scope = ['bathroom_02', 'bedroom_04', 'kitchen_02', 'living_room_08']
    TASK_LIST = {
                    'bathroom_02'    : ['26', '37', '43', '53', '69'],
                    'bedroom_04'     : ['134', '264', '320', '384', '387'],
                    'kitchen_02'     : ['90', '136', '157', '207', '329'],
                    'living_room_08' : ['92', '135', '190', '228', '254']
    }
'''
model = ActorCritic().to(device)
checkpoint = torch.load('./model/%s.pth' % (scope))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

NUM_EVAL_EPISODES = 100
MAX_STEP = 500
ep_lengths = []
min_length = []

for i in range(NUM_EVAL_EPISODES):
    for task_scope in task_scopes:
        env = Environment({'scene_name': scene_scope, 'terminal_state_id': int(task_scope)})
        state, target, min_dist = env.reset()
        state = torch.tensor(state).to(device)
        target = torch.tensor(target).to(device)
        min_length.append(min_dist)
        episode_length = 0
        for step in range(MAX_STEP):
            episode_length += 1
            with torch.no_grad():
                value, logit = model(state, target)
            prob = F.softmax(logit, dim=-1)
            action = prob.multinomial(num_samples=1).detach()
            state, reward, done, current_state_id, collide = env.step(action[0, 0])
            print('current_state_id:', current_state_id)
            state = torch.tensor(state).to(device)
            if done:
                break
        ep_lengths.append(episode_length)


num_fail = 0
for jj in range(NUM_EVAL_EPISODES):
    if ep_lengths[jj] == 500:
        num_fail = num_fail + 1
SR = 1 - num_fail / NUM_EVAL_EPISODES
print('SR:', SR * 100)



SPL = 0
for ii in range(NUM_EVAL_EPISODES):
    if ep_lengths[ii] != 500:
        SPL = SPL + min_length[ii] / ep_lengths[ii]
SPL = SPL / NUM_EVAL_EPISODES * 100
print('SPL:', SPL)

