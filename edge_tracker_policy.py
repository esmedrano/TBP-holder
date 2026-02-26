# CHANGED: Added DistScanFullPolicy for quick, full object scanning 
class DistScanFullPolicy(InformedPolicy):
    """
    This is a dynamic policy for quick, full object scanning that uses distant-agent 
    edge finding for each camera location. 
    """

    def __init__(
        self,
        grid_offset: float = 0.5,
        look_up_degrees: float = 5.0,
        reverse_degrees: float = 1.0,
        scan_up_degrees: float = 1.0,
        scan_left_degrees: float = 1.0,
        scan_down_degrees: float = 1.0,
        scan_right_degrees: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.grid_offset = grid_offset
        self.look_up_degrees = look_up_degrees
        self.reverse_degrees = reverse_degrees
        self.scan_up_degrees = scan_up_degrees
        self.scan_left_degrees = scan_left_degrees
        self.scan_down_degrees = scan_down_degrees
        self.scan_right_degrees = scan_right_degrees
        self._latest_patch_observation = None
        self.latest_semantic_3d = None
        self.latest_depth = None
        self._phase = "look_up"

        self.grid_points = None 
        self.jump_angle = None
        self.image_shape = None
        self.grid_midpoint = None
        self.previous_phase_counter = 0

        # Stuck phase vars
        self.last_direction = None
        self.stuck_jumps_counter = 0
        self.queued_action = None

        # DEBUG
        self.target_location = [0, 0, 0]

    def pre_episode(self):
        super().pre_episode()
        self._latest_patch_observation = None
        self.latest_semantic_3d = None
        self.latest_depth = None
        self._phase = "look_up"

        # Stuck phase vars
        self.last_direction = None
        self.stuck_jumps_counter = 0
        self.queued_action = None

        # E NOTE: These resets may not be needed but are less compute intensive then setting every step
        self.grid_points = None 
        self.image_shape = None
        self.grid_midpoint = None
        self.jump_angle = None
        self.previous_phase_counter = 0
    
    def build_grid (self):
        self.image_shape = self.latest_depth.shape

        self.grid_mid_point = [int(self.image_shape[0] / 2), int(self.image_shape[1] / 2)]

        self.grid_points = [
            [int(self.grid_mid_point[0] * (1 - self.grid_offset)), int(self.grid_mid_point[1])], 
            #[int(self.grid_mid_point[0] * (1 - self.grid_offset)), int(self.grid_mid_point[1] * (1 - self.grid_offset))],

            [int(self.grid_mid_point[0]), int(self.grid_mid_point[1] * (1 - self.grid_offset))],
            #[int(self.grid_mid_point[0] * (1 + self.grid_offset)), int(self.grid_mid_point[1] * (1 - self.grid_offset))],

            [int(self.grid_mid_point[0] * (1 + self.grid_offset)), int(self.grid_mid_point[1])],
            #[int(self.grid_mid_point[0] * (1 + self.grid_offset)), int(self.grid_mid_point[1] * (1 + self.grid_offset))],

            [int(self.grid_mid_point[0]), int(self.grid_mid_point[1] * (1 + self.grid_offset))],
            #[int(self.grid_mid_point[0] * (1 - self.grid_offset)), int(self.grid_mid_point[1] * (1 + self.grid_offset))]
        ]

    def set_patch_observation(self, observation: Mapping[str, Any] | None) -> None:
        self._latest_patch_observation = observation
        if observation is None:
            self.latest_semantic_3d = None
            self.latest_depth = None
            return

        self.latest_semantic_3d = observation.get("semantic_3d")
        self.latest_depth = observation.get("depth")
   
    def calculate_jump_angle(self):
        # Calculate the angle to jump based on which grid points are on the object
        # and their distance from the center point. 
        
        # E NOTE: Assuming a 90 degree field of view for the camera. A 90 FOV is also used in transforms.py DepthTo3DLocations()
        hfov = np.pi / 2
        opposite = self.image_shape[0] / 2
        focal_length = opposite / np.tan(hfov / 2)  
        distance_of_grid_point_from_center = abs(self.grid_points[0][0] - self.grid_mid_point[0])
        jump_radians = np.arctan(distance_of_grid_point_from_center / focal_length)
        # self.jump_angle = np.degrees(jump_radians)
        self.jump_angle = np.degrees(jump_radians) / 10  # DIVIDE BY THE ZOOM FACTOR!!!
    
    def check_grid_on_object(self) -> list:
        sem3d_obs_image = self.latest_semantic_3d.reshape(
            (self.latest_depth.shape[0], self.latest_depth.shape[1], 4)
        )
        on_object_image = sem3d_obs_image[:, :, 3]

        # Test the grid placement
        # on_object_image[self.grid_points[0][0]][self.grid_points[0][1]] = 919 
        # on_object_image[self.grid_points[1][0]][self.grid_points[1][1]] = 929
        # on_object_image[self.grid_points[2][0]][self.grid_points[2][1]] = 939
        # on_object_image[self.grid_points[3][0]][self.grid_points[3][1]] = 949
        # on_object_image[self.grid_points[4][0]][self.grid_points[4][1]] = 959
        # on_object_image[self.grid_points[5][0]][self.grid_points[5][1]] = 969 
        # on_object_image[self.grid_points[6][0]][self.grid_points[6][1]] = 979 
        # on_object_image[self.grid_points[7][0]][self.grid_points[7][1]] = 989
        # with open('output.txt', 'w') as f:
        #     for row in on_object_image:
        #         f.write(' '.join(map(str, row)) + '\n')
        return [
            on_object_image[self.grid_points[i][0]][self.grid_points[i][1]] for i in [*range(0, 4)]  # non inclusive 8
        ]
        
    def dynamic_call(self, state: MotorSystemState | None = None) -> Action | None:
        if self.processed_observations is None:
            return

        center_on_object = bool(self.processed_observations.get_on_object()) 

        # Raw obs are injected by graph_matching._pass_input_obs_to_motor_system
        # via self.set_patch_observation(...)
        if self.latest_depth is None or self.latest_semantic_3d is None:
            return
                # Build the grid and calculate the jump angle
        # E NOTE: Only build the grid and calculate the jump angle once per episode 
        if self.grid_points is None:
            self.build_grid()
        if self.jump_angle is None:
            self.calculate_jump_angle()
        grid_mapping = self.check_grid_on_object() 

        # Debugging movement
        print("\nPhase: ", self._phase)
        print("   ", grid_mapping[0], "   ")  
        print(grid_mapping[1], "0.0" if not center_on_object else "1.0", grid_mapping[3])  
        print("   ", grid_mapping[2], "   ")  
        print(self.previous_phase_counter)

        # Debugging angle calculation
        # sem3d_obs_image = self.latest_semantic_3d.reshape(
        #     (self.latest_depth.shape[0], self.latest_depth.shape[1], 4)
        # )
        # reshaped_locations = sem3d_obs_image[:, :, :3]
        # center_location = reshaped_locations[self.grid_mid_point[0], self.grid_mid_point[1]] if reshaped_locations is not None else None
        # grid_point_locations = [reshaped_locations[self.grid_points[i][0], self.grid_points[i][1]] if reshaped_locations is not None else None for i in range(8)]
        # print("\nPhase: ", self._phase)
        # print("Last center location: ", center_location)
        # print("Grid locations: ", grid_point_locations)
        # print("Jump angle: ", self.jump_angle)
        # print("differnece: ", [center_location[i] - self.target_location[i] if center_location is not None and self.target_location is not None else None for i in range(3)])

        # Movement Logic 

        # Move to top of object
        if self._phase == "look_up":
            if center_on_object:
                return LookUp(
                    agent_id=self.agent_id,
                    rotation_degrees=self.look_up_degrees,
                )
            else:
                self._phase = "reverse_to_on"
                return LookDown(
                    agent_id=self.agent_id,
                    rotation_degrees=self.reverse_degrees,
                )
            
        if self._phase == "reverse_to_on":
            if not center_on_object:
                return LookDown(
                    agent_id=self.agent_id,
                    rotation_degrees=self.reverse_degrees,
                )
            else: 
                self._phase = "scan_left"
        
        # Stuck phase
        actions = {
            "left": TurnLeft(
                agent_id=self.agent_id,
                rotation_degrees=self.jump_angle,
            ),
            "down": LookDown(
                agent_id=self.agent_id,
                rotation_degrees=self.jump_angle,
            ),
            "right": TurnRight(
                agent_id=self.agent_id,
                rotation_degrees=self.jump_angle,
            ),
            "up": LookUp(
                agent_id=self.agent_id,
                rotation_degrees=self.jump_angle,
            ),
        }

        reverse_actions = {
            "left": TurnRight(
                agent_id=self.agent_id,
                rotation_degrees=self.jump_angle,
            ),
            "down": LookUp(
                agent_id=self.agent_id,
                rotation_degrees=self.jump_angle,
            ),
            "right": TurnLeft(
                agent_id=self.agent_id,
                rotation_degrees=self.jump_angle,
            ),
            "up": LookDown(
                agent_id=self.agent_id,
                rotation_degrees=self.jump_angle,
            )
        }

        direction_to_grid_mapping = {
            "left": 1,
            "down": 2,
            "right": 3,
            "up": 0,
        }

        phases = ["scan_left", "scan_down", "scan_right", "scan_up"]

        if not center_on_object and self._phase != "look_up" and self._phase != "reverse_to_on":
            # print("Last direction: ", self.last_direction)
            # print("Queued action: ", self.queued_action)
            self.stuck_jumps_counter += 1
            # Reverse twice if stuck for the third time
            if self.stuck_jumps_counter > 2:  
                self.queued_action = reverse_actions[self.last_direction]
                # print("Reverse: ", reverse_actions[self.last_direction])
                return reverse_actions[self.last_direction]
            # Only jump if the target is on object
            if grid_mapping[direction_to_grid_mapping[self.last_direction]] > 0:
                # print("Jump: ", actions[self.last_direction])
                return actions[self.last_direction]
            # Otherwise switch phases 
            else:
                self.stuck_jumps_counter = 0
                # Set this true so that the regular algo works.
                center_on_object = True
                # Increment the phase with list wrap around
                current_phase_index = phases.index(self._phase)
                next_phase = phases[(current_phase_index + 1) % len(phases)]
                self._phase = next_phase
                # print("Next phase: ", self._phase)
        # Reset counter just in case last move was in stuck phase and it is no longer stuck now
        else:
            self.stuck_jumps_counter = 0

        if self.queued_action != None:
            # Pass action and reset queued_action
            action = self.queued_action
            self.queued_action = None

            # Increment the phase with list wrap around
            current_phase_index = phases.index(self._phase)
            next_phase = phases[(current_phase_index + 1) % len(phases)]
            self._phase = next_phase
            return action
        
        # Regular movement
        if self._phase == "scan_left":
            if center_on_object and grid_mapping[0] > 0:  
                self.previous_phase_counter += 1 
                self.last_direction = "up"
                if self.previous_phase_counter > 1:
                    self._phase = "scan_up"
                    self.previous_phase_counter = 0
                return LookUp(
                    agent_id=self.agent_id,
                    rotation_degrees=self.jump_angle,
                )
            elif center_on_object and grid_mapping[1] > 0:  
                self.previous_phase_counter = 0
                # Debugging angle calculation
                # self.target_location = reshaped_locations[self.grid_points[2][0], self.grid_points[2][1]]
                self.last_direction = "left"
                return TurnLeft(
                    agent_id=self.agent_id,
                    rotation_degrees=self.jump_angle,
                )
            elif center_on_object and grid_mapping[1] == 0:  
                # Reset phase counter during phase change ??
                self._phase = "scan_down"
                self.last_direction = "no_direction"
                return
                # WARNING: don't jump here becuase there is a chance it is off object. Check during the next phase
        
        if self._phase == "scan_down":
            if center_on_object and grid_mapping[1] > 0: 
                self.previous_phase_counter += 1 
                self.last_direction = "left"
                if self.previous_phase_counter > 1:
                    self._phase = "scan_left"
                    self.previous_phase_counter = 0
                return TurnLeft(
                    agent_id=self.agent_id,
                    rotation_degrees=self.jump_angle,
                )
            elif center_on_object and grid_mapping[2] > 0:  
                # print("down")
                self.previous_phase_counter = 0
                self.last_direction = "down"
                return LookDown(
                    agent_id=self.agent_id,
                    rotation_degrees=self.jump_angle,
                )
            elif center_on_object and grid_mapping[2] == 0:  
                # print("right")
                self._phase = "scan_right"
                self.last_direction = "no_direction"
                # WARNING: don't jump here becuase there is a chance it is off object. Check during the next phase
        
        if self._phase == "scan_right":
            if center_on_object and grid_mapping[2] > 0:  
                # print("down")
                self.previous_phase_counter += 1
                self.last_direction = "down"
                if self.previous_phase_counter > 1:
                    self._phase = "scan_down"
                    self.previous_phase_counter = 0
                return LookDown(
                    agent_id=self.agent_id,
                    rotation_degrees=self.jump_angle,
                )
            elif center_on_object and grid_mapping[3] > 0:  
                # phase counter reseting is important because if the right phase goes down once and doesn't change phases,
                # then it will think it went down twice next time it only needs to go down once, which
                # will cause it to go back to the down phase, then back to the left phase, and then in circles.
                self.previous_phase_counter = 0
                self.last_direction = "right"
                return TurnRight(
                    agent_id=self.agent_id,
                    rotation_degrees=self.jump_angle,
                )
            elif center_on_object and grid_mapping[3] == 0: 
                # print("up")
                self._phase = "scan_up"
                self.last_direction = "no_direction"
                return 
                # WARNING: don't jump here becuase there is a chance it is off object. Check during the next phase
            
        if self._phase == "scan_up":
            if center_on_object and grid_mapping[3] > 0:  
                self.previous_phase_counter += 1 
                self.last_direction = "right"
                if self.previous_phase_counter > 1:
                    self._phase = "scan_right"
                    self.previous_phase_counter = 0
                return TurnRight(
                    agent_id=self.agent_id,
                    rotation_degrees=self.jump_angle,
                )
            elif center_on_object and grid_mapping[0] > 0:  
                self.previous_phase_counter = 0
                self.last_direction = "up"
                return LookUp(
                    agent_id=self.agent_id,
                    rotation_degrees=self.jump_angle,
                )
            elif center_on_object and grid_mapping[0] == 0: 
                self._phase = "scan_left"
                self.last_direction = "no_direction"
                return 
                # WARNING: don't jump here becuase there is a chance it is off object. Check during the next phase
