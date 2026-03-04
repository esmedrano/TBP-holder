# CHANGED: Added DistScanFullPolicy for quick, full object scanning 
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
        target_object_id: int | None = None,
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
        self.target_object_id = target_object_id
        self._latest_patch_observation = None
        self.latest_semantic_3d = None
        self.latest_depth = None
        self._phase = "look_up"

        self.center_on_object = None
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
        self.min_max_left_displacment = [0, 0]
        self.min_max_down_displacment = [0, 0]
        self.min_max_right_displacment = [0, 0]
        self.min_max_up_displacment = [0, 0]
        
        # Stuck phase vars
        self.last_direction = None
        self.stuck_jumps_counter = 0
        self.queued_action = None
        self.stuck = False
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
        self.direction_to_grid_mapping = {
            "left": 1,
            "down": 2,
            "right": 3,
            "up": 0,
        }
        self.phases = ["scan_left", "scan_down", "scan_right", "scan_up"]

        # Interior phase vars
        self.max_interior_passes = 4
        self.total_y_degrees = None
        self.jump_down_angle = None
        self.boundary_rotation_counters = []
        self.jumped_down = False
        self.turning_right = True
        self.turning_left = False
        self.jump_counter = 0

        # Translation phase vars
        self.translation_phase = None

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
        self.on_object_image = []

        # Boundary end condition vars
        self.rotation_counter = [0, 0]
        self.sem3d_obs_image = None

        # Interior phase vars
        self.total_y_degrees = None
        self.jump_down_angle = None
        self.boundary_rotation_counters = []
        self.jumped_down = False
        self.turning_right = True
        self.turning_left = False
        self.jump_counter = 0

        # perspective translation state (used by translate_to_new_perspective)
        self.translation_phase = None

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
            if not self.center_on_object:
                return LookDown(
                    agent_id=self.agent_id,
                    rotation_degrees=self.reverse_degrees,
                )
            else: 
                self.starting_world_coord = self.sem3d_obs_image[self.img_center_index[0], self.img_center_index[1]]
                self._phase = "scan_left"

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

    def calculate_displacment(self):
        # Calculate displacement for each step
        self.current_location = self.sem3d_obs_image[self.img_center_index[0], self.img_center_index[1]]
        displacement = math.sqrt(
            (self.current_location[0] - self.starting_world_coord[0]) ** 2 +
            (self.current_location[1] - self.starting_world_coord[1]) ** 2 +
            (self.current_location[2] - self.starting_world_coord[2]) ** 2
        )
        if self.last_direction == "left":
            if displacement < self.min_max_left_displacment[0]:
                self.min_max_left_displacment[0] = displacement
            if displacement > self.min_max_left_displacment[1]:
                self.min_max_left_displacment[1] = displacement
        if self.last_direction == "down":
            if displacement < self.min_max_down_displacment[0]:
                self.min_max_down_displacment[0] = displacement
            if displacement > self.min_max_down_displacment[1]:
                self.min_max_down_displacment[1] = displacement
        if self.last_direction == "right":
            if displacement < self.min_max_right_displacment[0]:
                self.min_max_right_displacment[0] = displacement
            if displacement > self.min_max_right_displacment[1]:
                self.min_max_right_displacment[1] = displacement
        if self.last_direction == "up":
            if displacement < self.min_max_up_displacment[0]:
                self.min_max_up_displacment[0] = displacement
            if displacement > self.min_max_up_displacment[1]:
                self.min_max_up_displacment[1] = displacement
        
        print("current displacement: ", displacement)
        print(
            "minmax displacements: ", 
            self.min_max_left_displacment,
            self.min_max_down_displacment,
            self.min_max_right_displacment,
            self.min_max_up_displacment,
        )

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
                # Reset phase counter during phase change ??
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
                # print("right")
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
                # print("up")
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
                self._phase = "scan_left"
                self.last_direction = "no_direction"
                self.boundary_rotation_counters.append(self.rotation_counter)
                return 
                # WARNING: don't jump here becuase there is a chance it is off object. Check during the next phase 

    def check_if_object_boundary_complete(self):
        self.step_count += 1
        # Check if self._phase is not None or else the interior scan phase will be set to "position_left" for every
        # interior scan step
        if self.step_count > 10 and self._phase is not None:
            if (self.rotation_counter[0] == 0 and   
                self.rotation_counter[1] == 0 and
                self.rotation_counter[0] == 0 and 
                self.rotation_counter[1] == 0):
                print("end_condition")
                self.interior_phase = "position_left"
                self._phase = None

    def interior_scan(self):
        if self._phase is None and self.interior_phase is not None:
            # Calculate the jump_down angle 
            # Do this once per episode
            if self.total_y_degrees is None:
                self.total_y_degrees = (self.largest_counter[1][1] - self.smallest_counter[1][1]) * self.jump_angle
                self.jump_down_angle = self.total_y_degrees / self.max_interior_passes
            
            # reposition to top left
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
            if self.jump_counter >= self.max_interior_passes:
                self.interior_phase = None
                self.translation_phase = "start"

            # Now that the camera has positioned to the top left boundary, scan the interior
            if self.interior_phase == "start_scan":
                print("start scan phase")
                if all(self.rotation_counter[i] >= self.largest_counter[i][i] for i in (0, 1)):
                    pass
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
                        self.jump_counter += 1
                        return LookDown(
                            agent_id=self.agent_id,
                            rotation_degrees=self.jump_down_angle,
                        )
                    if self.turning_left:
                        self.turning_right = True
                        self.turning_left = False
                        self.jump_counter += 1
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
                    self.jump_counter += 1
                    return LookDown(
                        agent_id=self.agent_id,
                        rotation_degrees=self.jump_down_angle,
                    )

    def rotate_to_new_perspective(self):
        """Shift the camera to a slightly different yaw while keeping overlap and
        maintaining the same distance from the object.

        The idea is to perform a small leftward rotation about the vertical (y)
        axis and then correct the agent's forward/backward position if the depth
        to the object at the image centre has changed.  A two‑step state machine
        is used so that we can inspect the depth before and after the rotation.

        This ensures the new viewpoint contains a few pixels of the previous view 
        and the agent does not drift closer or further from the object as it 
        traverses around it. The method returns a single action per call; 
        successive calls handle the rotation and the correction.
        """  
        return self.rotate_target_object(0, 15, 0)
        
    def rotate_target_object(
        self,
        x_degrees: float = 0.0,
        y_degrees: float = 0.0,
        z_degrees: float = 0.0,
        relative: bool = True,
    ) -> RotateObject:
        """Build an action that rotates the configured object around x/y/z axes.

        Args:
            x_degrees: Rotation around +x axis (left-right axis).
            y_degrees: Rotation around +y axis (up axis).
            z_degrees: Rotation around +z axis (forward axis).
            relative: If True, apply incrementally; otherwise set absolute rotation.

        Returns:
            RotateObject action that can be returned by the policy.

        Notes:
            `target_object_id` may be None in unsupervised runs. In that case,
            simulator runtime resolves the object by semantic_id (if available)
            or by single-object fallback.
        """

        qx = qt.from_rotation_vector([np.deg2rad(x_degrees), 0.0, 0.0])
        qy = qt.from_rotation_vector([0.0, np.deg2rad(y_degrees), 0.0])
        qz = qt.from_rotation_vector([0.0, 0.0, np.deg2rad(z_degrees)])
        delta_q = qz * qy * qx
        delta_q_wxyz = tuple(qt.as_float_array(delta_q))
        semantic_id = getattr(self, "_target_semantic_id", None)
        semantic_id_int = int(semantic_id) if semantic_id is not None else None

        return RotateObject(
            agent_id=self.agent_id,
            object_id=self.target_object_id,
            rotation_quat=delta_q_wxyz,
            semantic_id=semantic_id_int,
            relative=relative,
        )

    def dynamic_call(self, state: MotorSystemState | None = None) -> Action | None:
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
        The camera rotates by the same ammount for each direction (up, left, down, right), so the current offset
        can be stored using a counter for the x and y directions that incrememnts by one for each movement.
        These are used to set the boudaries of the interior scan.
        The stored min max values must be updated with the current rotation counter values if the new vlaues are smaller / larger
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
            self.calculate_displacment()

        # Track the edge of the object; the helper returns an action when a
        # movement is required, otherwise None.  
        edge_action = self.track_edge()
        if edge_action is not None:
            return edge_action

        # After the full object boundary has been traversed, the edge phase will be
        # set to None and the camera will begin to scan the interior of the silhouette  
        interior_movement = self.interior_scan()
        if interior_movement is not None and self._phase is None:
            return interior_movement
        
        translation_movement = self.rotate_to_new_perspective()
        print("translation phase", self.translation_phase)
        print("movement: ", translation_movement)
        if translation_movement is not None and self.translation_phase == "start":
            return translation_movement
