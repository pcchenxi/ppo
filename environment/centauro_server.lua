package.path=package.path .. ";/home/xi/workspace/ppo/environment/?.lua"
require("common_functions")
require("ompl_functions")
require("robot_control")

-- simSetThreadSwitchTiming(2) 
-- simExtRemoteApiStart(19999)

function get_minimum_obs_dist(inInts,inFloats,inStrings,inBuffer)
    local threshold = 0.1
    local res, data = simCheckDistance(_collection_robot_hd, _collection_hd, threshold)
    if data == nil then 
        dist = threshold
    else 
        dist = data[7]
    end
    -- print(dist)    
    return {}, {dist}, {}, ''
end

function reset(inInts,inFloats,inStrings,inBuffer)
    local radius = inFloats[1]
    init(radius, 1)
    return {}, {}, {}, ''
end

function step(inInts,inFloats,inStrings,inBuffer)
    -- print('step')
    res = do_action_rl(_robot_hd, inFloats)
    -- sample_obstacle_position(obs_hds, #obs_hds)

    return {}, {}, {}, res
end

function move_robot(inInts,inFloats,inStrings,inBuffer)
    -- print('step')
    local robot_pos = simGetObjectPosition(_robot_hd, -1)
    robot_pos[1] =  inFloats[1]
    robot_pos[2] =  inFloats[2]

    simSetObjectPosition(_robot_hd, -1, robot_pos)
    -- sample_obstacle_position(obs_hds, #obs_hds)
    return {}, {}, {}, ''
end

function get_obstacle_info(inInts,inFloats,inStrings,inBuffer)

    local obs_info = {}
    for i=1, #_obstacle_dynamic_hds, 1 do 
        local pos = simGetObjectPosition(_obstacle_dynamic_hds[i], -1)
        local res, type, dim = simGetShapeGeomInfo(_obstacle_dynamic_hds[i])
        obs_info[#obs_info+1] = pos[1]
        obs_info[#obs_info+1] = pos[2]

        obs_info[#obs_info+1] = dim[1]
        obs_info[#obs_info+1] = dim[2]
        obs_info[#obs_info+1] = dim[3]

        print('shape: ', dim[1], dim[2], dim[3], dim[4])
    end

    return {}, obs_info, {}, ''
end

function get_robot_state(inInts,inFloats,inStrings,inBuffer)
    local target_pos =simGetObjectPosition(_target_hd, _robot_hd)
    local target_ori =simGetObjectPosition(_target_hd, _robot_hd)

    local pos =simGetObjectPosition(_robot_hd,-1)
    local ori =simGetObjectQuaternion(_robot_hd,-1)
    -- local joint_pose = get_joint_values(_joint_hds)
    local leg_l = get_current_l(_robot_hd)

    -- x, y, theta, h, l,   tx, ty, t_theta,   t_h, t_l
    local state = {}
    local target_angle = math.atan2(target_pos[1], target_pos[2])
    local target_dist = math.sqrt(target_pos[1]*target_pos[1] + target_pos[2]*target_pos[2])
    state[1] = target_pos[1]-- target_angle
    state[2] = target_pos[2] --target_dist
    state[3] = target_ori[3]
    state[4] = target_pos[3] - 0.4
    state[5] = _pre_target_l

    -- state[1] = pos[1]
    -- state[2] = pos[2]
    -- state[3] = ori[3]
    -- state[4] = pos[3]
    -- state[5] = leg_l

    -- state[6] = target_pos[1]
    -- state[7] = target_pos[2]
    -- state[8] = target_ori[3]

    -- state[9] = 0.0
    -- state[10] = 0.1

    for i=1, #_obstacle_dynamic_hds, 1 do 
        local obs_pos = simGetObjectPosition(_obstacle_dynamic_hds[i], _robot_hd)
        local obs_pos_global = simGetObjectPosition(_obstacle_dynamic_hds[i], -1)
        -- local res, type, dim = simGetShapeGeomInfo(obstacle_dynamic_hds[i])
        
        local x = math.abs(obs_pos_global[1])
        local y = math.abs(obs_pos_global[2])

        if x < 2.5 and y < 2.5 then   
            local obs_angle = math.atan2(obs_pos[1], obs_pos[2])
            local obs_dist = math.sqrt(obs_pos[1]*obs_pos[1] + obs_pos[2]*obs_pos[2])        
            if obs_dist > 1 then
                obs_dist = -1
                obs_angle = -1
            end 
            state[#state+1] = obs_angle
            state[#state+1] = obs_dist
            state[#state+1] = obs_pos_global[3] 
        end
    end

    state[#state+1] = ori[3]
    state[#state+1] = pos[3]
    state[#state+1] = leg_l

    state[#state+1] = 0.5 - math.abs(pos[1])
    state[#state+1] = 0.5 - math.abs(pos[2])

    -- print ('in get robot state:', #state[3])
    return {}, state, {}, ''
end

function generate_path()
    init_params(2, 8, 'centauro', 'obstacle_all', true)
    task_hd, state_dim = init_task('centauro','task_1')
    path = compute_path(task_hd, 10)
    print ('path found ', #path)
    -- displayInfo('finish 1 '..#path)

    for i=1, 30, 1 do 
        applyPath(task_hd, path, 0.1)
    end
    simExtOMPL_destroyTask(task_hd)

    return path
end

function applyPath(task_hd, path, speed)
    -- simSetModelProperty(robot_hd, 32)

    local state = {}
    for i=1,#path-state_dim,state_dim do
        for j=1,state_dim,1 do
            state[j]=path[i+j-1]
        end
        do_action_hl(_robot_hd, state)
        -- res = simExtOMPL_writeState(task_hd, state) 
        -- pos = {}
        -- pos[1] = state[1]
        -- pos[2] = state[2]
        -- pos[3] = 0
        -- print (pos[1])
        -- simSetObjectPosition(robot_hd, -1, pos)
        -- sleep (0.005)
        sleep(speed)
        simSwitchThread()
    end
    -- simSetModelProperty(robot_hd, 0)
end

function start()
    -- sleep (3)
    -- print('reset')
    _fake_robot_hd = simGetObjectHandle('fake_robot')
    _robot_hd = simGetObjectHandle('centauro')
    _robot_body_hd = simGetObjectHandle('body_ref')
    _target_hd = simGetObjectHandle('target')
    _joint_hds = get_joint_hds(16)

    _start_pos = simGetObjectPosition(_robot_hd, -1)
    _start_ori = simGetObjectQuaternion(_robot_hd,-1)
    _start_joint_values = get_joint_values(_joint_hds)

    _start_t_pos = simGetObjectPosition(_target_hd, -1)
    _start_t_ori = simGetObjectQuaternion(_target_hd,-1)

    _collection_hd = simGetCollectionHandle('obstacle_all')
    _collection_robot_hd = simGetCollectionHandle('centauro_mesh')

    _obstacles_hds = simGetCollectionObjects(_collection_hd)

    _obstacle_dynamic_collection = simGetCollectionHandle('obstacle_dynamic')
    _obstacle_dynamic_hds = simGetCollectionObjects(_obstacle_dynamic_collection)

    _pre_robot_pos = _start_pos
    _pre_robot_ori = _start_ori
    _pre_target_pos = _start_t_pos
    _pre_target_ori = _start_t_ori
    _pre_target_l = 0.1

    -- print (_start_pos[1], _start_pos[2])
end


function sample_obstacle_position()
    local v = 0.02
    local inside_obs_index = {}
    for i=1, #_obstacle_dynamic_hds, 1 do
        obs_pos = simGetObjectPosition(_obstacle_dynamic_hds[i], -1)

        local x = math.abs(obs_pos[1])
        local y = math.abs(obs_pos[2])

        local bound_x = 1
        local bound_y = 2
        if x < 2.5 and y < 2.5 then 
            inside_obs_index[#inside_obs_index +1] = i
            obs_pos[1] = (math.random()-0.5)*2 * bound_x 
            obs_pos[2] = (math.random()-0.5)*2 * bound_y 

            if obs_pos[1] > bound_x then
                obs_pos[1] = bound_x
            elseif obs_pos[1] < -bound_x then 
                obs_pos[1] = -bound_x
            end

            if obs_pos[2] > bound_y then
                obs_pos[2] = bound_y
            elseif obs_pos[2] < -bound_y then 
                obs_pos[2] = -bound_y
            end
        end
        -- print(obs_pos[1], obs_pos[2])
        simSetObjectPosition(_obstacle_dynamic_hds[i], -1, obs_pos)
    end
    return inside_obs_index
end

function sample_initial_poses(radius, resample)

    if resample == 0 then 
        simSetObjectPosition(_robot_hd, -1, _pre_robot_pos)
        simSetObjectQuaternion(_robot_hd, -1, _pre_robot_ori)
        simSetObjectPosition(_target_hd, -1, _pre_target_pos)
        simSetObjectQuaternion(_target_hd, -1, _pre_target_ori)
        return 0
    end

    if resample == 1 then 
        inside_obs_index = sample_obstacle_position()
    end 

    local robot_pos = {}
    robot_pos[1] = 0 --(math.random() - 0.5) * 2 * 0.5
    robot_pos[2] = 0 --(math.random() - 0.5) * 2 * 0.5
    robot_pos[3] = _start_pos[3]

    local robot_ori = {}
    robot_ori[1] = _start_ori[1]
    robot_ori[2] = _start_ori[2]
    robot_ori[3] = (math.random() - 0.5) *2 * math.pi    --_start_ori[3]
    robot_ori[4] = _start_ori[4]
    -- local res_robot = 0

    local target_pos = {}
    local target_ori = {} 
    if resample == -1 then 
        target_pos = _pre_target_pos
        target_ori = _pre_target_ori
    else 
        target_pos[1] = 0 --(math.random() - 0.5) *2 + robot_pos[1] --* 2 * 0.5
        target_pos[2] = math.random()
        target_pos[3] = (math.random() - 0.5) * 2 * 0.1 + 0.4

        target_ori[1] = _start_ori[1] 
        target_ori[2] = _start_ori[2]
        target_ori[3] = (math.random() - 0.5) * 2 * math.pi
        target_ori[4] = _start_ori[4]
    end

    simSetObjectPosition(_target_hd,-1,target_pos)
    simSetObjectQuaternion(_target_hd, -1, target_ori)

    _pre_robot_pos = robot_pos
    _pre_robot_ori = robot_ori
    _pre_target_pos = target_pos
    _pre_target_ori = target_ori
    _pre_target_l = (math.random() - 0.5) * 2 * 0.05 + 0.07


    -- ep type
    local type = 1
    if math.random() < 0.5 then 
        type = 2
    end 
    if type == 1 then 
        if #inside_obs_index > 0 then
            local obs_pos = {}
            local obs_index = math.random(#inside_obs_index)
            obs_index = inside_obs_index[obs_index]
            local obs_pos_before =  simGetObjectPosition(_obstacle_dynamic_hds[obs_index], -1)
            obs_pos[1] = (robot_pos[1] + target_pos[1])/2 + (math.random() - 0.5) * 0.3
            obs_pos[2] = (robot_pos[2] + target_pos[2])/2 + (math.random() - 0.5) * 0.5
            obs_pos[3] = obs_pos_before[3]
            simSetObjectPosition(_obstacle_dynamic_hds[obs_index], -1, obs_pos)
        end
    else 
        if #inside_obs_index > 0 then
            local obs_pos = {}
            local obs_index = math.random(#inside_obs_index)
            obs_index = inside_obs_index[obs_index]
            local obs_pos_before =  simGetObjectPosition(_obstacle_dynamic_hds[obs_index], -1)
            obs_pos[1] = (robot_pos[1] + target_pos[1])/2 - math.random() * 0.5
            obs_pos[2] = (robot_pos[2] + target_pos[2])/2 + (math.random() - 0.5) * 0.5
            obs_pos[3] = obs_pos_before[3]
            simSetObjectPosition(_obstacle_dynamic_hds[obs_index], -1, obs_pos)
            local obs_pos = {}
            local obs_index = math.random(#inside_obs_index)
            obs_index = inside_obs_index[obs_index]
            local obs_pos_before =  simGetObjectPosition(_obstacle_dynamic_hds[obs_index], -1)
            obs_pos[1] = (robot_pos[1] + target_pos[1])/2 + math.random() * 0.5
            obs_pos[2] = (robot_pos[2] + target_pos[2])/2 + (math.random() - 0.5) * 0.5
            obs_pos[3] = obs_pos_before[3]
            simSetObjectPosition(_obstacle_dynamic_hds[obs_index], -1, obs_pos)         
        end
    end

    simSetObjectPosition(_robot_hd, -1, robot_pos)
    simSetObjectQuaternion(_robot_hd, -1, robot_ori)
    set_joint_values(_joint_hds, _start_joint_values)

    simSetObjectPosition(_fake_robot_hd,-1,robot_pos)
    simSetObjectQuaternion(_fake_robot_hd, -1, robot_ori)
    local res_robot = simCheckCollision(_fake_robot_hd, _collection_hd)



    simSetObjectPosition(_fake_robot_hd,-1,target_pos)
    simSetObjectQuaternion(_fake_robot_hd, -1, target_ori)
    local res_target = simCheckCollision(_fake_robot_hd, _collection_hd)

    if math.abs(target_pos[1]) > 2.5 or math.abs(target_pos[2]) > 2.5 then 
        res_target = 1
    end

    -- print (res_robot, res_target)
    return res_robot+res_target

end

function init(radius, resample)
    -- resample = 1
    -- global_counter = global_counter + 1
    -- if global_counter%300000 == 0 then 
    --     resample = 1
    -- end

    local init_value = 1
    while (init_value ~= 0) do
        init_value = sample_initial_poses(radius, resample)
    end

    -- print('reset')
    -- print(_start_pos[1], _start_pos[2], _start_pos[3])
    -- print(_start_joint_values[9], _start_joint_values[10], _start_joint_values[11], _start_joint_values[12])

    -- sleep(5)
    -- simSetModelProperty(_robot_hd, 32)
    -- g_path = generate_path()
end

initialized = false
global_counter = 0

-- get_obstacle_info(nil, nil, nil, nil)
start()
init(1.5, 1)
-- simSetModelProperty(_robot_hd, 32)

-- print(_start_pos[1], _start_pos[2], _start_ori[3])
-- for i=1, 10, 1 do
-- --     -- i = 0
--     action = {_start_pos[1], _start_pos[2], 0, 0, 0.1}
--     do_action_hl(_robot_hd, action)
--     simSwitchThread()
--     sleep(3)
-- end
-- start()

-- init()

-- for i=1, 5000, 1 do
--     action = {0,0,0,0,0,0,0,0}
--     for j = 1, 8, 1 do
--         action[j] = (math.random()-0.5)*2
--     end
--     -- action = {-1,1,-1,1,0,0,0,0}
--     print (do_action(robot_hd, action))
-- end


while simGetSimulationState()~=sim_simulation_advancing_abouttostop do
    -- do something in here
    -- simSwitchThread()
end




