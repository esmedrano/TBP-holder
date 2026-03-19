class DistScanFullPolicy(InformedPolicy):
    """
    This is a policy for quick, full object scanning that uses distant-agent 
    edge tracking for each camera perspective. 

    It traces the silhouette then fills it in.
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
        perspective_back_distance_world: float = 0.0,
        target_object_id: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.state = None
        self.grid_offset = grid_offset
        self.look_up_degrees = look_up_degrees
        self.reverse_degrees = reverse_degrees
        self.scan_up_degrees = scan_up_degrees
        self.scan_left_degrees = scan_left_degrees
        self.scan_down_degrees = scan_down_degrees
        self.scan_right_degrees = scan_right_degrees
        self.perspective_back_distance_world = perspective_back_distance_world
        self.target_object_id = target_object_id
        self._latest_patch_observation = None
        self.latest_semantic_3d = None
        self.latest_depth = None
        self._phase = "look_up"
        self.starting_world_coord = 0

        self.center_on_object = None
        self.img_center_index = None
        self.grid_mapping = None
        self.grid_points = None 
        self.jump_angle = None
        self.image_shape = None
        self.grid_midpoint = None
        self.previous_phase_counter = 0
        self.largest_counter = [[0, 0], [0, 0]]
        self.smallest_counter = [[0, 0], [0, 0]]
        self.interior_phase = ""
        self.step_count = 0

        # Boundary and end condition vars
        self.rotation_counter = [0, 0]
        self.sem3d_obs_image = None

        self.left_coord = None
        self.right_coord = None
        self.down_coord = None
        self.up_coord = None
        
        # Stuck phase vars
        self.last_direction = None
        self.stuck_jumps_counter = 0
        self.queued_action = None
        self.stuck = False
        self.actions = None
        self.direction_to_grid_mapping = {
            "left": 1,
            "down": 2,
            "right": 3,
            "up": 0,
        }
        self.phases = ["scan_left", "scan_down", "scan_right", "scan_up"]

        # Interior phase vars
        self.max_interior_passes = 1
        self.total_y_degrees = None
        self.jump_down_angle = None
        self.boundary_rotation_counters = []
        self.jumped_down = False
        self.turning_right = True
        self.turning_left = False
        self.jump_down_counter = 0
        self.main_counter_increment = 0

        # Translation phase vars
        self.translation_phase = None
        self.object_center_in_rotations = None
        self.rotation_counter_phase = True
        self.z_displacement = None
        self.object_origin = None
        self.camera_world_loc = None
        self.sensor_rot_world = None
        self.center_phase = "horizontal"
        self.vector_b = None
        self.vector_b_mag = None
        self.starting_point = None

        # DEBUG
        self.target_location = [0, 0, 0]
        self.debug_counter = 0

    def pre_episode(self):
        super().pre_episode()
        self.state = None
        
        self._latest_patch_observation = None
        # This is the raw observation, not to be confused with the reshaped observation "sem3d_obs_image" 
        # that has the dimensions of the camera view
        self.latest_semantic_3d = None  
        self.latest_depth = None
        self._phase = "look_up"
        self.starting_world_coord = 0

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
        self.on_object_image = []

        # Boundary end condition vars
        self.rotation_counter = [0, 0]
        self.sem3d_obs_image = None

        self.left_coord = None
        self.right_coord = None
        self.down_coord = None
        self.up_coord = None

        # Interior phase vars
        self.total_y_degrees = None
        self.jump_down_angle = None
        self.boundary_rotation_counters = []
        self.jumped_down = False
        self.turning_right = True
        self.turning_left = False
        self.jump_down_counter = 0
        self.main_counter_increment = 0

        # perspective translation state (used by translate_to_new_perspective)
        self.translation_phase = None
        self.object_center_in_rotations = None
        self.rotation_counter_phase = True
        self.z_displacement = None
        self.object_origin = None
        self.camera_world_loc = None
        self.sensor_rot_world = None
        self.center_phase = "horizontal"
        self.vector_b = None
        self.vector_b_mag = None
        self.starting_point = None

    def build_grid (self):
        self.img_center_index = [self.latest_depth.shape[0] // 2, self.latest_depth.shape[1] // 2]

        self.grid_points = [
            [int(self.img_center_index[0] * (1 - self.grid_offset)), int(self.img_center_index[1])], 
            #[int(self.img_center_index[0] * (1 - self.grid_offset)), int(self.img_center_index[1] * (1 - self.grid_offset))],

            [int(self.img_center_index[0]), int(self.img_center_index[1] * (1 - self.grid_offset))],
            #[int(self.img_center_index[0] * (1 + self.grid_offset)), int(self.img_center_index[1] * (1 - self.grid_offset))],

            [int(self.img_center_index[0] * (1 + self.grid_offset)), int(self.img_center_index[1])],
            #[int(self.img_center_index[0] * (1 + self.grid_offset)), int(self.img_center_index[1] * (1 + self.grid_offset))],

            [int(self.img_center_index[0]), int(self.img_center_index[1] * (1 + self.grid_offset))],
            #[int(self.img_center_index[0] * (1 - self.grid_offset)), int(self.img_center_index[1] * (1 + self.grid_offset))]
        ]

    def calculate_jump_angle(self):
        # Calculate the angle to jump based on which grid points are on the object
        # and their distance from the center point. 
        
        # E NOTE: Assuming a 90 degree field of view for the camera. A 90 FOV is also used in transforms.py DepthTo3DLocations()
        hfov = np.pi / 2 
        opposite = self.latest_depth.shape[0] / 2
        focal_length = opposite / np.tan(hfov / 2)  
        distance_of_grid_point_from_center = abs(self.grid_points[0][0] - self.img_center_index[0])
        jump_radians = np.arctan(distance_of_grid_point_from_center / focal_length)
        # self.jump_angle = np.degrees(jump_radians)
        self.jump_angle = np.degrees(jump_radians) / 10  # DIVIDE BY THE ZOOM FACTOR!!!

        # Debugging angle calculation
        # sem3d_obs_image = self.latest_semantic_3d.reshape(
        #     (self.latest_depth.shape[0], self.latest_depth.shape[1], 4)
        # )
        # reshaped_locations = sem3d_obs_image[:, :, :3]
        # center_location = reshaped_locations[self.img_center_index[0], self.img_center_index[1]] if reshaped_locations is not None else None
        # grid_point_locations = [reshaped_locations[self.grid_points[i][0], self.grid_points[i][1]] if reshaped_locations is not None else None for i in range(4)]
        # print("\nPhase: ", self._phase)
        # print("Last center location: ", center_location)
        # print("Grid locations: ", grid_point_locations)
        # print("Jump angle: ", self.jump_angle)
        # print("differnece: ", [center_location[i] - self.target_location[i] if center_location is not None and self.target_location is not None else None for i in range(3)])

    def set_patch_observation(self, observation: Mapping[str, Any] | None) -> None:
        self._latest_patch_observation = observation
        if observation is None:
            self.latest_semantic_3d = None
            self.latest_depth = None
            return

        self.latest_semantic_3d = observation.get("semantic_3d")
        self.latest_depth = observation.get("depth")
   
    def check_grid_on_object(self) -> list:
        self.sem3d_obs_image = self.latest_semantic_3d.reshape(
            (self.latest_depth.shape[0], self.latest_depth.shape[1], 4)
        )
        on_object_image = self.sem3d_obs_image[:, :, 3]

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
        
        # This reduces the total pixel count of the image by one to avoid querrying index 64 of 63 when the grid points are on the edge of the image
        return [
            on_object_image[self.grid_points[i][0]][self.grid_points[i][1]] for i in [*range(0, 4)] 
        ]

    def move_cam_to_top(self):
        if self._phase == "look_up":
            if self.center_on_object:
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
            if not self.center_on_object or not self.grid_mapping[0]:
                return LookDown(
                    agent_id=self.agent_id,
                    rotation_degrees=self.reverse_degrees,
                )
            else: 
                self.starting_world_coord = self.sem3d_obs_image[self.img_center_index[0], self.img_center_index[1]]
                self.rotation_counter = [0, 0]
                self._phase = "scan_left"
                self.step_count = 0
                self.previous_phase_counter = 0
                self.smallest_counter = [[0, 0], [0, 0]]
                self.largest_counter = [[0, 0], [0, 0]]
                self.jump_down_counter = 0
                self.main_counter_increment = 0

                self.left_coord = None
                self.right_coord = None
                self.down_coord = None
                self.up_coord = None

                self.total_y_degrees = None
                self.jump_down_angle = None
                self.jumped_down = False
                self.turning_right = True
                self.turning_left = False

                self.object_center_in_rotations = None
                self.rotation_counter_phase = True
                self.z_displacement = None
                self.object_origin = None
                self.camera_world_loc = None
                self.sensor_rot_world = None
                self.center_phase = "horizontal"
                self.vector_b = None
                self.vector_b_mag = None
                self.starting_point = None

    def compute_min_max_rotation_vals(self):
        print("current rotation: ", self.rotation_counter)
        for axis in [*range(0, len(self.rotation_counter))]:
            if self.rotation_counter[axis] < self.smallest_counter[axis][axis]:
                small_counter_copy = self.rotation_counter.copy()
                self.smallest_counter[axis] = small_counter_copy
            if self.rotation_counter[axis] > self.largest_counter[axis][axis]:
                large_counter_copy = self.rotation_counter.copy()
                self.largest_counter[axis] = large_counter_copy
        print("smallest and largest counters: ", self.smallest_counter, self.largest_counter)

    def check_if_stuck(self): 
        if self.actions == None:
            self.actions = {
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
            self.reverse_actions = {
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

        if not self.center_on_object and self._phase != "look_up" and self._phase != "reverse_to_on" and self._phase != None:
            self.stuck = True
            # print("Last direction: ", self.last_direction)
            # print("Queued action: ", self.queued_action)
            self.stuck_jumps_counter += 1
            # Reverse twice if stuck for the third time
            if self.stuck_jumps_counter > 2:  
                self.queued_action = self.reverse_actions[self.last_direction]
                # print("Reverse: ", self.reverse_actions[self.last_direction])
                return self.reverse_actions[self.last_direction]
            # Only jump if the target is on object
            if self.grid_mapping[self.direction_to_grid_mapping[self.last_direction]] > 0:
                # print("Jump: ", actions[self.last_direction])
                return self.actions[self.last_direction]
            # Otherwise switch self.phases 
            else:
                self.stuck_jumps_counter = 0
                # Set this true so that the regular algo works.
                self.center_on_object = True
                # Increment the phase with list wrap around
                current_phase_index = self.phases.index(self._phase)
                next_phase = self.phases[(current_phase_index + 1) % len(self.phases)]
                self._phase = next_phase
                # print("Next phase: ", self._phase)
        # Reset counter just in case last move was in stuck phase and it is no longer stuck now
        else:
            self.stuck_jumps_counter = 0

        if self.stuck and self.queued_action != None:
            # Pass action and reset queued_action
            action = self.queued_action
            self.queued_action = None
            self.stuck = False

            # Increment the phase with list wrap around
            current_phase_index = self.phases.index(self._phase)
            next_phase = self.phases[(current_phase_index + 1) % len(self.phases)]
            self._phase = next_phase
            return action

    def find_furthest_coords(self):
        """Find the furthest world coordinates in every direction. 
        In the simulation, the world coordinates are fixed, but in the real world, there is a chance that the camera itself is repositioned
        such that the x or even y directions are flipped. To account for this, every direction is treated as positive by taking the absolute
        value of the current location and the last stored value, then another conditional checks the direction of the movement so that the 
        correct coordinate is updated.  
        """
        self.current_location = self.sem3d_obs_image[self.img_center_index[0], self.img_center_index[1]]

        if (self.left_coord is None or abs(self.current_location[0]) > abs(self.left_coord[0])) and self.last_direction == "left":
            self.left_coord = self.current_location.copy()

        if (self.right_coord is None or abs(self.current_location[0]) > abs(self.right_coord[0])) and self.last_direction == "right":
            self.right_coord = self.current_location.copy()

        if (self.up_coord is None or abs(self.current_location[1]) > abs(self.up_coord[1])) and self.last_direction == "up":
            self.up_coord = self.current_location.copy()

        if (self.down_coord is None or abs(self.current_location[1]) > abs(self.down_coord[1])) and self.last_direction == "left":
            self.down_coord = self.current_location.copy()

        print("coordinates: ", self.left_coord, self.right_coord, self.down_coord, self.up_coord) 

    def track_edge(self):
        if self._phase == "scan_left":
            if self.center_on_object and self.grid_mapping[0] > 0:  
                self.previous_phase_counter += 1 
                self.last_direction = "up"
                if self.previous_phase_counter > 1:
                    self._phase = "scan_up"
                    self.previous_phase_counter = 0
                self.boundary_rotation_counters.append(self.rotation_counter)
                self.rotation_counter[1] += 1 
                return LookUp(
                    agent_id=self.agent_id,
                    rotation_degrees=self.jump_angle,
                )
            elif self.center_on_object and self.grid_mapping[1] > 0:  
                self.previous_phase_counter = 0
                # Debugging angle calculation
                # self.target_location = reshaped_locations[self.grid_points[2][0], self.grid_points[2][1]]
                self.last_direction = "left"
                self.rotation_counter[0] -= 1 
                return TurnLeft(
                    agent_id=self.agent_id,
                    rotation_degrees=self.jump_angle,
                )
            elif self.center_on_object and self.grid_mapping[1] == 0:  
                self.previous_phase_counter = 0
                self._phase = "scan_down"
                self.last_direction = "no_direction"
                self.boundary_rotation_counters.append(self.rotation_counter)
                return
                # WARNING: don't jump here becuase there is a chance it is off object. Check during the next phase
        
        if self._phase == "scan_down":
            if self.center_on_object and self.grid_mapping[1] > 0: 
                self.previous_phase_counter += 1 
                self.last_direction = "left"
                if self.previous_phase_counter > 1:
                    self._phase = "scan_left"
                    self.previous_phase_counter = 0
                self.boundary_rotation_counters.append(self.rotation_counter)
                self.rotation_counter[0] -= 1
                return TurnLeft(
                    agent_id=self.agent_id,
                    rotation_degrees=self.jump_angle,
                )
            elif self.center_on_object and self.grid_mapping[2] > 0:  
                # print("down")
                self.previous_phase_counter = 0
                self.last_direction = "down"
                self.rotation_counter[1] -= 1
                return LookDown(
                    agent_id=self.agent_id,
                    rotation_degrees=self.jump_angle,
                )
            elif self.center_on_object and self.grid_mapping[2] == 0:  
                self.previous_phase_counter = 0
                self._phase = "scan_right"
                self.last_direction = "no_direction"
                self.boundary_rotation_counters.append(self.rotation_counter)
                return
                # WARNING: don't jump here becuase there is a chance it is off object. Check during the next phase
        
        if self._phase == "scan_right":
            if self.center_on_object and self.grid_mapping[2] > 0:  
                # print("down")
                self.previous_phase_counter += 1
                self.last_direction = "down"
                if self.previous_phase_counter > 1:
                    self._phase = "scan_down"
                    self.previous_phase_counter = 0
                self.boundary_rotation_counters.append(self.rotation_counter)
                self.rotation_counter[1] -= 1
                return LookDown(
                    agent_id=self.agent_id,
                    rotation_degrees=self.jump_angle,
                )
            elif self.center_on_object and self.grid_mapping[3] > 0:  
                # phase counter reseting is important because if the right phase goes down once and doesn't change self.phases,
                # then it will think it went down twice next time it only needs to go down once, which
                # will cause it to go back to the down phase, then back to the left phase, and then in circles.
                self.previous_phase_counter = 0
                self.last_direction = "right"
                self.rotation_counter[0] += 1
                return TurnRight(
                    agent_id=self.agent_id,
                    rotation_degrees=self.jump_angle,
                )
            elif self.center_on_object and self.grid_mapping[3] == 0: 
                self.previous_phase_counter = 0
                self._phase = "scan_up"
                self.last_direction = "no_direction"
                self.boundary_rotation_counters.append(self.rotation_counter)
                return 
                # WARNING: don't jump here becuase there is a chance it is off object. Check during the next phase
            
        if self._phase == "scan_up":
            if self.center_on_object and self.grid_mapping[3] > 0:  
                self.previous_phase_counter += 1 
                self.last_direction = "right"
                if self.previous_phase_counter > 1:
                    self._phase = "scan_right"
                    self.previous_phase_counter = 0
                self.boundary_rotation_counters.append(self.rotation_counter)
                self.rotation_counter[0] += 1
                return TurnRight(
                    agent_id=self.agent_id,
                    rotation_degrees=self.jump_angle,
                )
            elif self.center_on_object and self.grid_mapping[0] > 0:  
                self.previous_phase_counter = 0
                self.last_direction = "up"
                self.rotation_counter[1] += 1
                return LookUp(
                    agent_id=self.agent_id,
                    rotation_degrees=self.jump_angle,
                )
            elif self.center_on_object and self.grid_mapping[0] == 0: 
                self.previous_phase_counter = 0
                self._phase = "scan_left"
                self.last_direction = "no_direction"
                self.boundary_rotation_counters.append(self.rotation_counter)
                return 
                # WARNING: don't jump here becuase there is a chance it is off object. Check during the next phase 

    def check_if_object_boundary_complete(self):
        self.step_count += 1

        # reverse_moves = ["up", "right", "down", "left"]
        # last_move_index = reverse_moves.index(self.last_direction)
        # indeces = [(last_move_index + i) % 4 for i in [*range(0, 4)]]
        # if self.last_four_moves == [reverse_moves[i] for i in indeces]:
        #     pass

        # Check if self._phase is not None or else the interior scan phase will be set to "position_left" for every
        # interior scan step
        if self.step_count > 10 and self._phase is not None:
            if (self.rotation_counter[0] == 0 and   
                self.rotation_counter[1] == 1):
                self.interior_phase = "position_left"
                self._phase = None

    def interior_scan(self):
        if self._phase is None and self.interior_phase is not None:
            # Calculate the jump_down angle 
            # Do this once per silhouette perspective
            if self.total_y_degrees is None:
                self.total_y_degrees = (self.largest_counter[1][1] - self.smallest_counter[1][1]) * self.jump_angle
                self.jump_down_angle = self.total_y_degrees / self.max_interior_passes
                self.main_counter_increment = self.jump_down_angle / self.jump_angle
            
            # Reposition to top left
            # This uses the min max rotation counters, so it can still use the same counters to get back to the origin, or anywhere else on the object
            if self.interior_phase == "position_left":
                reposition_x_degrees = (self.rotation_counter[0] - self.smallest_counter[0][0]) * self.jump_angle
                self.interior_phase = "position_up"
                self.rotation_counter[0] = self.smallest_counter[0][0]
                return TurnLeft(
                    agent_id=self.agent_id,
                    rotation_degrees=reposition_x_degrees,
                )
            if self.interior_phase == "position_up":
                reposition_y_degrees = (self.largest_counter[1][1] - self.rotation_counter[1]) * self.jump_angle
                self.interior_phase = "start_scan"
                self.rotation_counter[1] = self.largest_counter[1][1]
                return LookUp(
                    agent_id=self.agent_id,
                    rotation_degrees=reposition_y_degrees,
                )
            
            # Check if the entire object has been scanned
            # This should be done by checking if the all downward jumps have been completed
            # because if it is done by checking the rotation counters there is a chance the 
            # y direction may be slightly off... ;)
            if self.jump_down_counter >= self.max_interior_passes:
                self.interior_phase = None
                self.translation_phase = "start"

            # Now that the camera has positioned to the top left boundary, scan the interior
            # The interior scan can be sped up by limiting the total number of horizontal scans. This changes the jump down angle between 
            # each scan, which unaligns the camera from the counter grid it could have used to get back to the origin / center. 
            # To account for this, the main_counter_increment is calculated above as: 
            # self.main_counter_increment = self.jump_down_angle / self.jump_angle
            # and subtracted from the original rotation counter y value. The original counter can now be used to recenter the object in
            # self.rotate_to_new_perspective()
            
            # I attempted to use world coordinates to recenter the object after scanning (in self.rotate_to_new_perspective()), but I would 
            # need to create a bounding box in world coordinates for any rotation of the world axes. This proves problematic as it requires 
            # a linear transform for each camera rotation to check just how far it is traveling along the shifted world axes??  
            if self.interior_phase == "start_scan":
                print("scanning")
                if self.rotation_counter[0] < self.largest_counter[0][0] and self.turning_right:
                    self.rotation_counter[0] += 1
                    return TurnRight(
                        agent_id=self.agent_id,
                        rotation_degrees=self.jump_angle,
                    )
                elif not self.jumped_down:
                    self.jumped_down = True
                    if self.turning_right:
                        self.turning_right = False
                        self.turning_left = True
                        self.jump_down_counter += 1
                        self.rotation_counter[1] -= self.main_counter_increment
                        return LookDown(
                            agent_id=self.agent_id,
                            rotation_degrees=self.jump_down_angle,
                        )
                    if self.turning_left:
                        self.turning_right = True
                        self.turning_left = False
                        self.jump_down_counter += 1
                        self.rotation_counter[1] -= self.main_counter_increment
                        return LookDown(
                            agent_id=self.agent_id,
                            rotation_degrees=self.jump_down_angle,
                        )
                elif self.rotation_counter[0] > self.smallest_counter[0][0] and self.turning_left:
                    self.rotation_counter[0] -= 1
                    return TurnLeft(
                        agent_id=self.agent_id,
                        rotation_degrees=self.jump_angle,
                    ) 
                else: 
                    self.jumped_down = False
                    self.turning_right = True
                    self.turning_left = False
                    self.jump_down_counter += 1
                    self.rotation_counter[1] -= self.main_counter_increment
                    return LookDown(
                        agent_id=self.agent_id,
                        rotation_degrees=self.jump_down_angle,
                    )

    def rotate_to_new_perspective(self):
        """ Rotate the object as if being moved by a hand so that a new silhouette can be collected
        """ 

        """ Calculate object depth adjustmant
        If you really want to understand this, draw a picture, and consider deriving the dot product from the law of cosines.
        And also, maybe derive the cross product from the dot product 
        And maybe the Rodrigues formula: v' = vcos(θ) + (n x v)sin(θ) + n(n•v)(1 - cos(θ)) 
        And then check out Grant Sanderson's and Ben Eater's quaternions visualizer at https://eater.net/quaternions

        The displacement is from the world coordinate furthest to the left of the camera's perspective "P", to the object's rotation axis origin "Ac". 
        The object is rotated around the world y axis "A", so A needs to be rotated so that it is parallel to the camera's y axis "Ac". 
        Without the rotation, the displacement will be inaccurate. 

        When passed to habitat sim, the displacement is automatically applied along the camera axes, 
        but before being passed, it still needs to be calcualted using the camera's axes: 

        The shortest distance D from point P to axis Ac is a line perpendicular to Ac.
        Ac = the camera's rotation relative to the world, which must be calculated by multiplying the agent's quaternion
        relative to the world "Aq" by the camera's quaternion relative to the agent "Aa" (at least I think that is the case).

        Ac = Aa * Aq
        |Ac| = 1
        
        The vector V from the point P to the origin of Ac "O"
        V = P - O

        The magnitude of V
        |V| = sqrt(sum(A[i] ** 2 for i in [0, 1, 2]))
        
        The magnitude of V parallel. It's the adjacent side of the right triangle fromed by V and Ac  
        |V| parallel = V dot Ac
        
        The magnitude of V perpendicular. It's the opposite side of the right triangle fromed by V and Ac
        |V| perpendicular = sqrt(|V| ** 2 - |V| parallel ** 2)
        """
        if self.z_displacement == None:
            agent_state = self.state[self.agent_id]
            agent_pos_world = np.asarray(agent_state["position"], dtype=float)
            agent_rot_world = agent_state["rotation"]
            sensor_key = (
                "patch_0.depth"
                if "patch_0.depth" in agent_state["sensors"]
                else next(k for k in agent_state["sensors"] if k.endswith(".depth"))
            )
            sensor_pos_rel_agent = np.asarray(
                agent_state["sensors"][sensor_key]["position"], dtype=float
            )
            sensor_rot_agent = agent_state["sensors"][sensor_key]["rotation"]
            self.sensor_rot_world = agent_rot_world * sensor_rot_agent
            self.camera_world_loc = agent_pos_world + qt.rotate_vectors(self.sensor_rot_world, sensor_pos_rel_agent)
            camera_displacement = math.sqrt(sum((self.camera_world_loc[i] - self.starting_world_coord[i]) ** 2 for i in [0, 1, 2]))

            # Debugging the camera displacement calculations: 
            print("\nAgent state: ", agent_state)
            print("Sensor state: ", agent_state["sensors"])
            print("Agent pos rel world: ", agent_pos_world)
            print("Agent rot rel world: ", agent_rot_world)
            print("Sensor pos rel agent: ", sensor_pos_rel_agent)
            print("Sensor rot rel agent: ", sensor_rot_agent)
            print("Sensor rotation rel world: ", self.sensor_rot_world)
            print("Camera loc rel world = agent world loc: ", self.camera_world_loc)
            print("Camera displacement: ", camera_displacement)

            self.object_origin = np.asarray(agent_state["object_origin"], dtype=float)
            world_y_direction = [0, 1, 0]
            cam_rot_unit_vector = qt.rotate_vectors(self.sensor_rot_world, world_y_direction)
            furthest_point_vector = [self.left_coord[i] - self.object_origin[i] for i in [0, 1, 2]]
            furthest_point_vector_mag = math.sqrt(sum(furthest_point_vector[i] ** 2 for i in [0, 1, 2]))
            parallel_component_mag = sum(furthest_point_vector[i] * cam_rot_unit_vector[i] for i in [0, 1, 2])
            perpendicular_component_mag = math.sqrt(furthest_point_vector_mag ** 2 - parallel_component_mag ** 2)
            leftmost_coord_d_from_origin = perpendicular_component_mag
            
            # Debugging the object displacement calculation 
            print("\nObject origin p:", self.object_origin)
            print("Camera rotation applied to world +Y:", cam_rot_unit_vector)
            print("Furthest point vector: ", furthest_point_vector)
            print("Furthest point vector magnitude: ", furthest_point_vector_mag)
            print("Parallel component magnitude: ", parallel_component_mag)
            print("Perpendicular component magnitude: ", perpendicular_component_mag)
            print("Displacement: ", self.z_displacement)

            starting_point_d_from_origin = math.sqrt(sum((self.camera_world_loc[i] - self.starting_world_coord[i])** 2 for i in [0, 1, 2]))

            self.z_displacement = starting_point_d_from_origin - leftmost_coord_d_from_origin

        """Originally I thought the object needed to be centered, but instead the camera center should be 
        aligned with the rotation axis origin. In this case it is defined in the 3D model, but in the real 
        world it will depend on the location of the hand and will need to be calculated. Once the camera 
        and rotation axis are aligned, the object will still be centered after the rotation. 
        
        I'll leave the original centering method that uses the rotation counter commented out, just in case 
        it proves useful later. 
        
        The phase trigger is start becasue that is what it was set to in order to enter this function call, 
        and it hasn't been changed to anything else yet. It could also be set to "center"
        """
        if self.translation_phase == "start":
            """# The rotation counter centering method
            # Calculate the x, y center of the object once per episode
            print("Current location", self.current_location)
            if self.object_center_in_rotations == None:
                object_center_in_rotations = [
                    (self.largest_counter[0][0] + self.smallest_counter[0][0]) / 2, 
                    (self.largest_counter[1][1] + self.smallest_counter[1][1]) / 2
                ]

            # To end the "start" phase, rotate the camera as close as possible to the x, y center of the object using the rotation counter.
            bounding_box_threshold = 1

            if (self.rotation_counter[0] <= object_center_in_rotations[0] and not 
                all([self.rotation_counter[0] < object_center_in_rotations[0] + bounding_box_threshold and 
                    self.rotation_counter[0] > object_center_in_rotations[0] - bounding_box_threshold])):
                self.rotation_counter[0] += 1
                self.last_direction = "right"
                print("centering right")
                return TurnRight(agent_id=self.agent_id, rotation_degrees=self.jump_angle)

            if (self.rotation_counter[0] >= object_center_in_rotations[0] and not 
                all([self.rotation_counter[0] < object_center_in_rotations[0] + bounding_box_threshold and 
                    self.rotation_counter[0] > object_center_in_rotations[0] - bounding_box_threshold])):
                self.rotation_counter[0] -= 1
                self.last_direction = "left"
                print("centering left")
                return TurnLeft(agent_id=self.agent_id, rotation_degrees=self.jump_angle)

            if (self.rotation_counter[1] <= object_center_in_rotations[1] and not 
                all([self.rotation_counter[1] < object_center_in_rotations[1] + bounding_box_threshold and 
                    self.rotation_counter[1] > object_center_in_rotations[1] - bounding_box_threshold])):
                self.rotation_counter[1] += 1 
                self.last_direction = "up"
                print("centering up")
                return LookUp(agent_id=self.agent_id, rotation_degrees=self.jump_angle)

            if (self.rotation_counter[1] >= object_center_in_rotations[1] and not 
                all([self.rotation_counter[1] < object_center_in_rotations[1] + bounding_box_threshold and 
                    self.rotation_counter[1] > object_center_in_rotations[1] - bounding_box_threshold])):
                self.rotation_counter[1] -= 1
                self.last_direction = "down"
                print("centering down")
                return LookDown(agent_id=self.agent_id, rotation_degrees=self.jump_angle)
            """
            
            # The camera will be at either the bottom left or bottom right of the object. The goal is to align the camera center with 
            # the object origin, which can be accomplished by a horizontal and vertical camera rotation. 
            if self.center_phase == "horizontal":
                if self.rotation_counter[0] > 0: 
                    self.starting_point = self.right_coord
                elif self.rotation_counter[0] < 0:
                    self.starting_point = self.left_coord
                vector_a = [self.camera_world_loc[i] - self.starting_point[i] for i in [0, 1, 2]]  # This is h mag not p mag
                vector_a_mag = math.sqrt(sum(vector_a[i] ** 2 for i in [0, 1, 2]))
                starting_point_parallel_y_mag = self.starting_point[1] 
                point_b = [self.object_origin[0], starting_point_parallel_y_mag, self.object_origin[2]]
                self.vector_b = [self.camera_world_loc[i] - point_b[i] for i in [0, 1, 2]]
                self.vector_b_mag = math.sqrt(sum(self.vector_b[i] ** 2 for i in [0, 1, 2]))
                horizontal_angle_to_center = np.degrees(math.acos(sum(vector_a[i] * self.vector_b[i] for i in [0, 1, 2]) / (vector_a_mag * self.vector_b_mag)))
                
                self.center_phase = "vertical"
                
                # The object origin is wherever the rotation axis passes through the object, and the rotation counter origin is the 
                # world coordinate of the first observation during the track edge phase. 
                # The expected behavior is that if the rotation counter is negative, the camera should use the angle to move right
                # towards the rotation counter origin, but the goal is to move towards the object origin. There is a chance the 
                # rotation origin is as far left as it can get. In that case, the camera should move left. To check if this is the 
                # case, use the smallest x rotation counter. Do the same for the case in which the rotation origin is furthest to 
                # the right. 
                # Otherwise the final two elifs treat the rotation counter origin as if it is the object origin. It is important to 
                # use the rotation counters because there is a chance the camera axes move such that the world x axes have reversed 
                # direction, but the rotation counter direcions are always true to the camera's perspective.     
                # .....
                # This didn't feel necessary to me at first either, but I drew pictures. 
                # It feels like I should be able to do everything using only world coordinates....
                if self.smallest_counter[0] == 0 and self.rotation_counter[0] == 0:        
                    return TurnRight(agent_id=self.agent_id, rotation_degrees=horizontal_angle_to_center)
                elif self.largest_counter[0] == 0 and self.rotation_counter[0] == 0:
                    return TurnLeft(agent_id=self.agent_id, rotation_degrees=horizontal_angle_to_center)
                elif self.rotation_counter[0] < 0:
                        return TurnRight(agent_id=self.agent_id, rotation_degrees=horizontal_angle_to_center)
                elif self.rotation_counter[0] > 0:
                    return TurnLeft(agent_id=self.agent_id, rotation_degrees=horizontal_angle_to_center)

            if self.center_phase == "vertical":
                # This is great for centering the camera vertically on the object origin, but for habitat sim, this is not garunteed to be the center of the object
                # Leave this block for later when the object origin height is centered after calculating from hand placement. 
                # Otherwise use the rotation counters to center vertically.   
                """
                vector_c = [self.camera_world_loc[i] - self.object_origin[i] for i in [0, 1, 2]]
                vector_c_mag = math.sqrt(sum(vector_c[i] ** 2 for i in [0, 1, 2]))
                point_d = [self.object_origin[0], self.down_coord[1], self.object_origin[2]]
                vector_d = [self.camera_world_loc[i] - point_d[i] for i in [0, 1, 2]]  # This should always be the down coord bc of how the interior phase works
                vector_d_mag = math.sqrt(sum(vector_d[i] ** 2 for i in [0, 1, 2]))
                vertical_angle_to_center = np.degrees(math.acos(sum(vector_c[i] * vector_d[i] for i in [0, 1, 2]) / (vector_c_mag * vector_d_mag)))                
                self.center_phase = None
                return LookUp(agent_id=self.agent_id, rotation_degrees=vertical_angle_to_center)
                """

                # Rotation counter vertical centering is copy pasted from the original centering block above.
                if self.object_center_in_rotations == None:
                    self.object_center_in_rotations = [
                        (self.largest_counter[0][0] + self.smallest_counter[0][0]) / 2, 
                        (self.largest_counter[1][1] + self.smallest_counter[1][1]) / 2
                    ]

                bounding_box_threshold = 1
                
                if (self.rotation_counter[1] < self.object_center_in_rotations[1] and not 
                    all([self.rotation_counter[1] < self.object_center_in_rotations[1] + bounding_box_threshold and 
                        self.rotation_counter[1] > self.object_center_in_rotations[1] - bounding_box_threshold])):
                    self.rotation_counter[1] += 1 
                    self.last_direction = "up"
                    print("centering up")
                    return LookUp(agent_id=self.agent_id, rotation_degrees=self.jump_angle)

                if (self.rotation_counter[1] > self.object_center_in_rotations[1] and not 
                    all([self.rotation_counter[1] < self.object_center_in_rotations[1] + bounding_box_threshold and 
                        self.rotation_counter[1] > self.object_center_in_rotations[1] - bounding_box_threshold])):
                    self.rotation_counter[1] -= 1
                    self.last_direction = "down"
                    print("centering down")
                    return LookDown(agent_id=self.agent_id, rotation_degrees=self.jump_angle)

            self.translation_phase = "adjust depth"
            return 

        # Adjust depth 
        if self.translation_phase == "adjust depth":
            # Transform the displacement along the world x axis to the camera x axis by rotating the displacement vector
            # This must needs be done here because the camera x axis was rotated during the centering stage 
            translation_world_vec = qt.rotate_vectors(
                self.sensor_rot_world,
                np.asarray([0.0, 0.0, self.z_displacement], dtype=float),
            )
            z_translation = tuple(float(v) for v in translation_world_vec)
            z = (0.0, 0.0, self.z_displacement)
            self.translation_phase = "rotate"
            return self.rotate_target_object(translation_world=z)

        # Perform rotation
        if self.translation_phase == "rotate":
            vector_m = [self.object_origin[i] - self.left_coord[i] for i in [0, 1, 2]]
            vector_m_mag = math.sqrt(sum(vector_m[i] ** 2 for i in [0, 1, 2]))
            point_n = [self.camera_world_loc[0], self.left_coord[1], self.camera_world_loc[2]]
            vector_n = [self.object_origin[i] - point_n[i] for i in [0, 1, 2]]
            vector_n_mag = math.sqrt(sum(vector_n[i] ** 2 for i in [0, 1, 2]))
            rotation_angle_deg = np.degrees(math.acos(sum(vector_m[i] * vector_n[i] for i in [0, 1, 2]) / (vector_m_mag * vector_n_mag)))
            # camera_y_world = qt.rotate_vectors(self.sensor_rot_world, np.asarray([0.0, 1.0, 0.0]))
            # delta_q = qt.from_rotation_vector(np.deg2rad(angle_deg) * camera_y_world)
            # delta_q_wxyz = tuple(qt.as_float_array(delta_q))
            # if self.total_angles >= 85:
            #     self.translation_phase = None
            #     return
            self.translation_phase = None
            self._phase = "look_up"
            return self.rotate_target_object(0, rotation_angle_deg, 0)
        

        # Adjust to object surface normal
        """
        There is a chance that the object has a wide corner in which case the camera should adjust 
        to be normal with the surface.  
        """
        
    def rotate_target_object(
        self,
        x_degrees: float = 0.0,
        y_degrees: float = 0.0,
        z_degrees: float = 0.0,
        rotation_quat: tuple[float, float, float, float] | None = None,
        translation_world: tuple[float, float, float] | None = None,
        relative: bool = True,
    ) -> RotateObject:
        """Build an action that rotates the configured object around x/y/z axes.

        Args:
            x_degrees: Rotation around +x axis (left-right axis).
            y_degrees: Rotation around +y axis (up axis).
            z_degrees: Rotation around +z axis (forward axis).
            relative: If True, apply incrementally; otherwise set absolute rotation.
            rotation_quat: Pre-built quaternion (w, x, y, z). When provided,
                x/y/z_degrees are ignored and this quaternion is used directly.
            translation_world: Optional world-coordinate translation delta `(x, y, z)`
                applied with the same action.

        Returns:
            RotateObject action that can be returned by the policy.

        Notes:
            `target_object_id` may be None in unsupervised runs. In that case,
            simulator runtime resolves the object by semantic_id (if available)
            or by single-object fallback.
        """

        if rotation_quat is not None:
            delta_q_wxyz = rotation_quat
        else:
            qx = qt.from_rotation_vector([np.deg2rad(x_degrees), 0.0, 0.0])
            qy = qt.from_rotation_vector([0.0, np.deg2rad(y_degrees), 0.0])
            qz = qt.from_rotation_vector([0.0, 0.0, np.deg2rad(z_degrees)])
            delta_q_wxyz = tuple(qt.as_float_array(qz * qy * qx))
        semantic_id = getattr(self, "_target_semantic_id", None)
        semantic_id_int = int(semantic_id) if semantic_id is not None else None

        return RotateObject(
            agent_id=self.agent_id,
            object_id=self.target_object_id,
            rotation_quat=delta_q_wxyz,
            semantic_id=semantic_id_int,
            relative=relative,
            translation_world=translation_world,
        )

    def translate_target_object_horizontal(
        self,
        direction: Literal["left", "right"],
        distance_world: float,
        *,
        relative: bool = True,
    ) -> RotateObject:
        """Build an action that translates the target object left or right.

        Args:
            direction: Horizontal image direction, either "left" or "right".
            distance_world: Translation magnitude in world units.
            relative: Whether to apply relative transform for the action.

        Returns:
            RotateObject action with identity rotation and world translation.

        Notes:
            Translation is aligned to camera horizontal axis when current policy
            state is available. If no sensor pose can be resolved, it falls back
            to world +x / -x.
        """
        magnitude = float(abs(distance_world))
        sign = 1.0 if direction == "right" else -1.0

        if self.state is not None:
            agent_state = self.state[self.agent_id]
            sensor_key = (
                "patch_0.depth"
                if "patch_0.depth" in agent_state["sensors"]
                else print("No depth agent")
            )
            agent_rot_world = agent_state["rotation"]
            sensor_rot_agent = agent_state["sensors"][sensor_key]["rotation"]
            sensor_rot_world = agent_rot_world * sensor_rot_agent
            camera_right_world = qt.rotate_vectors(
                sensor_rot_world,
                np.asarray([1.0, 0.0, 0.0], dtype=float),
            )
            translation_world_vec = sign * magnitude * np.asarray(
                camera_right_world,
                dtype=float,
            )
        else:
            translation_world_vec = np.asarray([sign * magnitude, 0.0, 0.0], dtype=float)

        translation_world = tuple(float(v) for v in translation_world_vec)
        return self.rotate_target_object(
            translation_world=translation_world,
            relative=relative,
        )

    def dynamic_call(self, state: MotorSystemState | None = None) -> Action | None:
        self.state = state
        print(self.grid_points)
        if self.processed_observations is None:
            return

        # Raw obs are injected by graph_matching._pass_input_obs_to_motor_system
        # via self.set_patch_observation(...)
        if self.latest_depth is None or self.latest_semantic_3d is None:
            return
        self.center_on_object = bool(self.processed_observations.get_on_object()) 
    
        # Build the grid and calculate the jump angle
        if self.grid_points is None:
            self.build_grid()
        if self.jump_angle is None:
            self.calculate_jump_angle()

        self.grid_mapping = self.check_grid_on_object()

        # angle_deg = 5
        # # camera_y_world = qt.rotate_vectors(self.sensor_rot_world, np.asarray([0.0, 1.0, 0.0]))
        # # delta_q = qt.from_rotation_vector(np.deg2rad(angle_deg) * camera_y_world)
        # # delta_q_wxyz = tuple(qt.as_float_array(delta_q))
        # if self.total_angles >= 150:
        #     self.translation_phase = None
        #     return
        # self.total_angles += angle_deg
        # return self.rotate_target_object(0, angle_deg, 0)

        # Debugging movement
        print("\nPhase: ", self._phase)
        print("   ", self.grid_mapping[0], "   ")  
        print(self.grid_mapping[1], "0.0" if not self.center_on_object else "1.0", self.grid_mapping[3])  
        print("   ", self.grid_mapping[2], "   ")  
        print(self.previous_phase_counter)

        # Move camera to the top of object to begin the edge tracking phase
        setup_action = self.move_cam_to_top()
        if setup_action is not None:
            return setup_action
        
        """
        The camera points by the same ammount for each direction (up, left, down, right), so the current offset
        can be stored using a counter for the x and y directions that incrememnts by one for each movement.
        These are used to set the boudaries of the interior scan.
        The stored min max values must be updated with the current rotation counter values if the new values are smaller / larger
        """
        self.compute_min_max_rotation_vals()
        
        # E TODO: this might be improved by checking if the current world coordinates are close to the starting point
        self.check_if_object_boundary_complete()
            
        # Attempt to resolve a stuck condition if the camera is ever off object.
        # If it produces an action then we should take it immediately. 
        stuck_action = self.check_if_stuck()
        if stuck_action is not None:
            return stuck_action

        # Calculate the displacement of the last jump and compare it to the stored min max vals 
        if self._phase is not None:
            self.find_furthest_coords()

        # Track the edge of the object; the helper returns an action when a
        # movement is required, otherwise None.  
        edge_action = self.track_edge()
        if edge_action is not None:
            return edge_action

        # After the full object boundary has been traversed, the edge phase will be
        # set to None and the camera will begin to scan the interior of the silhouette  
        if self._phase is None:
            interior_movement = self.interior_scan()
            if interior_movement is not None: 
                return interior_movement
        
        if self._phase is None:
            translation_movement = self.rotate_to_new_perspective()
            print("translation phase", self.translation_phase)
            if translation_movement is not None:
                return translation_movement
