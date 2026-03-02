## 配置
系统版本：ubuntu20
ROS版本：noetic
python版本：3.8
pytorch版本：2.1.0
cuda版本：12.3

## 项目结构
```
catkin_ws
        ---bulid
        ---devel
        ---src    
             ---crowd_behaviors    网上copy的用于控制仿真环境中的障碍物的生成与运动，用于借鉴，可忽略

             ---turtlebot3_description   
                     ----launch     用于启动代码
                           -----turtlebot3_crowd_sparse.launch  启动有障碍训练环境，并加载移动机器人
                           -----turtlebot3_crowd_none.launch   启动无障碍训练环境，并加载移动机器人
                           -----test_ddpg.launch   启动测试环境，并加载移动机器人
                           -----simulate_crowd.launch   启动训练环境中的障碍物运动
                           -----test_ddpg.launch           启动测试环境中的障碍物运动
                           
                     ----worlds  用于生成仿真环境
                           -----test_obstacle_20.world   测试环境
                           -----turtlebot3_crowd_sparse.world    有障碍训练环境
                           -----turtlebot3_crowd_none.world      无障碍训练环境

                     ----urdf   描述移动机器人的仿真模型
                   
                     ----scripts  控制仿真环境中障碍物的移动

                     ----meshes  模型渲染

            ---DDPG
                 ---configs  一些参数的设置
                 ---models  网络模型
                 ---results   结果输出
                 
                 ---ddpg.py   ddpg网络的构建
                 ---ddpg_lstm.py   ddpg_lstm网络的构建
                 ---ddpg_per.py     带有PER策略的ddpg网络构建
                 ---PER_buffer.py    PER策略的经验池构建
                 ---environment.py   强化学习要素的构建，与环境的交互
                 ---start_ddpg_training.py    开始对ddpg网络进行训练/测试
                 ---start_ddpg_lstm_training.py  开始对ddpg_lstm网络进行训练/测试
                 ---util.py    辅助函数
```          
**演示视频**
![静态演示](media/static.gif)
![动态演示](media/dynamic.gif)