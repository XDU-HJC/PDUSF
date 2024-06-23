import numpy as np
import torch
from network import ActorCritic
from scene_loader import THORDiscreteEnvironment as Environment
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('runs/loss_experiment')
scene = 'bathroom_02'
scene_scope_eval = 'bathroom_02'
scene_scope = 'bathroom_02'
task_scopes = ['26']

'''
    ls_scene_scope = ['bathroom_02', 'bedroom_04', 'kitchen_02', 'living_room_08']
    TASK_LIST = {
                    'bathroom_02'    : ['26', '37', '43', '53', '69'],
                    'bedroom_04'     : ['134', '264', '320', '384', '387'],
                    'kitchen_02'     : ['90', '136', '157', '207', '329'],
                    'living_room_08' : ['92', '135', '193', '228', '254']
    }
'''
lr = 0.0001
num_episodes = 5000
ep_len = 100
gamma = 0.99
return_list = []
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = ActorCritic().to(device)
# checkpoint = torch.load('./model/%s.pth' % (scene))
# model.load_state_dict(checkpoint['state_dict'])
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
Episode_length = []
iter = 0
for episode in range(num_episodes):
    for task_scope in task_scopes:
        env = Environment({'scene_name': scene_scope, 'terminal_state_id': int(task_scope)})
        state, target, _ = env.reset()
        state = torch.tensor(state).to(device)
        target = torch.tensor(target).to(device)

        episode_length = 0
        done = False
        while not done:
            values =[]
            states = []
            targets = []
            actions = []
            probs = []
            rewards = []
            for step in range(ep_len):
                episode_length += 1
                states.append(state)
                targets.append(targets)
                value, logit = model(state, target)
                prob = F.softmax(logit, dim=-1)
                action = prob.multinomial(num_samples=1).detach()

                state, reward, done, current_state_id, collide = env.step(action)
                done = done or episode_length >= 1000
                reward = max(min(reward, 1), -1)

                state = torch.tensor(state).to(device)


                values.append(value)
                actions.append(action)
                probs.append(prob)
                rewards.append(reward)

                if done:
                    break

            R = 0.0
            if not done:
                R, _ = model(state, target)

            values.reverse()
            rewards.reverse()
            actions.reverse()
            states.reverse()
            probs.reverse()

            batch_si = []
            batch_a = []
            batch_td = []
            batch_R = []
            batch_t = []


            for (ai, ri, vi, si, ti) in zip(actions, rewards, values, states, targets):
                R = ri + gamma * R
                td = R - vi
                a = np.zeros(4)
                a[ai] = 1

                batch_a.append(a)
                batch_R.append(R)
                batch_td.append(td)
                batch_si.append(si)
                batch_t.append(ti)

            batch_a = torch.tensor(batch_a).to(device)
            batch_R = torch.tensor(batch_R).to(device)
            batch_td = torch.tensor(batch_td).to(device)

            pi = torch.cat(probs, dim=0)
            log_pi = torch.log(torch.clamp(pi, 1e-20, 1.0))
            entroy = -torch.sum(pi * log_pi, dim=1)
            policy_loss = -torch.sum(torch.sum(log_pi * batch_a, dim=1) * batch_td + 0.01 * entroy)
            value_loss = 0.5 * torch.sum(batch_td)**2
            writer.add_scalar('value_loss', value_loss.item(), iter)
            writer.add_scalar('policy_loss', policy_loss.item(), iter)
            iter += 1

            optimizer.zero_grad()
            (policy_loss + 0.5 * value_loss).backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 50)
            optimizer.step()

        print('episode:', episode)
        print('epi_length:', episode_length)
        Episode_length.append(episode_length)




f = open("[Train] reward_%s.txt" % (scene), 'a')
f.truncate(0)
for i in range(len(Episode_length)):
    f.write('%s' % (Episode_length[i]))
    f.write('\n')

checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
torch.save(checkpoint, './model/%s.pth' % (scene))