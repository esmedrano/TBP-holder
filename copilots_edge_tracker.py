"""
This was Copilot's response when I asked it for an edge tracker that has been used by other projects. It uses OpenCV
to generate a list of x, y coordinates along the outer edge, ignoring any inner edges. It then calculates which point
is closest to the center of the grid. To find the target point it looks x indeces ahead of the closest point. 

It does not return the opposite action of the last action to avoid bouncing and instead looks further along the list a 
set number of times after which it defaults to the parent movement behavior. I think it starts bouncing becuase the edge
coordinate generator changes it's start point based on where the contour is in each frame? 

One edge case is that it travels a one pixel edge and then fails to back out of it using the opposite action, so the 
k counter allows enough retries to escape or allows the opposite action. (or at least it should, I think the opposite
action is still not allowed)

Another possible edge case is changing directions if the contour list is generated in the opposite direction as the 
last frame. I don't know how OpenCV generates it so I am going to stick to my algo that uses direction phases and 
data directly from the object. 
"""

class DistScanContourPolicy(InformedPolicy):
    def __init__(
        self,
        contour_area_threshold: float = 50.0,
        border_margin_px: int = 2,
        contour_lookahead: int = 8,
        reacquire_distance_px: float = 6.0,
        contour_direction: int = 1,
        deadband_px: int = 2,
        min_rotation_degrees: float = 0.5,
        max_rotation_degrees: float = 6.0,
        zoom_factor: float = 10.0,
        open_kernel_size: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.contour_area_threshold = contour_area_threshold
        self.border_margin_px = border_margin_px
        self.contour_lookahead = contour_lookahead
        self.reacquire_distance_px = reacquire_distance_px
        self.contour_direction = 1 if contour_direction >= 0 else -1
        self.deadband_px = deadband_px
        self.min_rotation_degrees = min_rotation_degrees
        self.max_rotation_degrees = max_rotation_degrees
        self.zoom_factor = zoom_factor
        self.open_kernel_size = open_kernel_size

        self._latest_patch_observation = None
        self.latest_semantic_3d = None
        self.latest_depth = None
        self._contour_idx = None
        self._phase = "contour_scan"
        self._last_contour_action = None
        self._last_contour_direction = None
        self._contour_repeat_k = 1

    def pre_episode(self):
        super().pre_episode()
        self._latest_patch_observation = None
        self.latest_semantic_3d = None
        self.latest_depth = None
        self._contour_idx = None
        self._phase = "contour_scan"
        self._last_contour_action = None
        self._last_contour_direction = None
        self._contour_repeat_k = 1

    def set_patch_observation(self, observation: Mapping[str, Any] | None) -> None:
        self._latest_patch_observation = observation
        if observation is None:
            self.latest_semantic_3d = None
            self.latest_depth = None
            return
        self.latest_semantic_3d = observation.get("semantic_3d")
        self.latest_depth = observation.get("depth")

    def _mask_from_semantic(self) -> np.ndarray | None:
        if self.latest_semantic_3d is None or self.latest_depth is None:
            return None
        h, w = self.latest_depth.shape
        sem3d_img = self.latest_semantic_3d.reshape((h, w, 4))
        mask = sem3d_img[:, :, 3] > 0
        return mask

    def _extract_contour_points(self, mask: np.ndarray) -> tuple[np.ndarray, float]:
        if cv2 is not None:
            mask_u8 = mask.astype(np.uint8)
            if self.open_kernel_size > 1:
                kernel = np.ones((self.open_kernel_size, self.open_kernel_size), dtype=np.uint8)
                mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                return np.zeros((0, 2), dtype=np.int32), 0.0
            contour = max(contours, key=cv2.contourArea)
            points = contour[:, 0, :].astype(np.int32)
            area = float(cv2.contourArea(contour))
            return points, area

        mask_clean = mask
        if self.open_kernel_size > 1:
            structure = np.ones((self.open_kernel_size, self.open_kernel_size), dtype=bool)
            mask_clean = scipy.ndimage.binary_opening(mask_clean, structure=structure)
        labeled, num_labels = scipy.ndimage.label(mask_clean)
        if num_labels == 0:
            return np.zeros((0, 2), dtype=np.int32), 0.0
        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0
        largest_label = int(np.argmax(component_sizes))
        mask_clean = labeled == largest_label
        mask_clean = scipy.ndimage.binary_fill_holes(mask_clean)
        boundary = mask_clean & ~scipy.ndimage.binary_erosion(mask_clean)
        rc_points = np.argwhere(boundary)
        if rc_points.size == 0:
            return np.zeros((0, 2), dtype=np.int32), 0.0
        center = rc_points.mean(axis=0)
        angles = np.arctan2(rc_points[:, 0] - center[0], rc_points[:, 1] - center[1])
        order = np.argsort(angles)
        rc_points = rc_points[order]
        xy_points = np.column_stack((rc_points[:, 1], rc_points[:, 0])).astype(np.int32)
        area = float(mask_clean.sum())
        return xy_points, area

    def _point_to_action(self, target_xy: np.ndarray, image_shape: tuple[int, int]) -> Action | None:
        h, w = image_shape
        cx, cy = w // 2, h // 2
        dx = int(target_xy[0]) - cx
        dy = int(target_xy[1]) - cy

        if abs(dx) <= self.deadband_px and abs(dy) <= self.deadband_px:
            return None

        hfov = np.pi / 2
        focal_x = (w / 2) / np.tan(hfov / 2)
        focal_y = (h / 2) / np.tan(hfov / 2)
        angle_x = np.degrees(np.arctan(abs(dx) / max(focal_x, 1e-6))) / max(self.zoom_factor, 1e-6)
        angle_y = np.degrees(np.arctan(abs(dy) / max(focal_y, 1e-6))) / max(self.zoom_factor, 1e-6)

        if abs(dx) >= abs(dy):
            rotation_degrees = float(np.clip(angle_x, self.min_rotation_degrees, self.max_rotation_degrees))
            if dx > 0:
                return TurnRight(agent_id=self.agent_id, rotation_degrees=rotation_degrees)
            return TurnLeft(agent_id=self.agent_id, rotation_degrees=rotation_degrees)

        rotation_degrees = float(np.clip(angle_y, self.min_rotation_degrees, self.max_rotation_degrees))
        if dy > 0:
            return LookDown(agent_id=self.agent_id, rotation_degrees=rotation_degrees)
        return LookUp(agent_id=self.agent_id, rotation_degrees=rotation_degrees)

    def _nearest_contour_idx(
        self,
        contour_points: np.ndarray,
        image_shape: tuple[int, int],
    ) -> tuple[int, float]:
        h, w = image_shape
        center_xy = np.array([w // 2, h // 2])
        d2 = np.sum((contour_points - center_xy) ** 2, axis=1)
        idx = int(np.argmin(d2))
        return idx, float(np.sqrt(d2[idx]))

    def _is_opposite_action(self, current: Action | None) -> bool:
        last = self._last_contour_action
        if last is None or current is None:
            return False

        opposite_pairs = (
            (LookUp, LookDown),
            (LookDown, LookUp),
            (TurnLeft, TurnRight),
            (TurnRight, TurnLeft),
        )
        return any(isinstance(last, a) and isinstance(current, b) for a, b in opposite_pairs)

    def dynamic_call(self, state: MotorSystemState | None = None) -> Action | None:
        if self.processed_observations is None:
            return

        mask = self._mask_from_semantic()
        if mask is None:
            return

        contour_points, area = self._extract_contour_points(mask)
        if contour_points.shape[0] == 0 or area < self.contour_area_threshold:
            print(
                f"Contour fallback: empty/too-small contour (n={contour_points.shape[0]}, area={area:.2f})"
            )
            return super().dynamic_call(state)

        h, w = mask.shape
        touches_border = (
            (contour_points[:, 0] <= self.border_margin_px).any()
            or (contour_points[:, 1] <= self.border_margin_px).any()
            or (contour_points[:, 0] >= (w - 1 - self.border_margin_px)).any()
            or (contour_points[:, 1] >= (h - 1 - self.border_margin_px)).any()
        )
        self._phase = "interior_scan" if not touches_border else "contour_scan"

        nearest_idx, nearest_distance = self._nearest_contour_idx(contour_points, (h, w))

        if nearest_distance > self.reacquire_distance_px:
            self._contour_idx = nearest_idx
        else:
            self._contour_idx = (
                nearest_idx + self.contour_direction * self.contour_lookahead
            ) % contour_points.shape[0]

        retries = self._contour_repeat_k + 1
        for attempt in range(retries):
            target_xy = contour_points[self._contour_idx]
            action = self._point_to_action(target_xy, (h, w))
            if action is not None and not self._is_opposite_action(action):
                self._last_contour_action = action
                current_direction = type(action).__name__
                if self._last_contour_direction == current_direction:
                    self._contour_repeat_k += 1
                else:
                    self._last_contour_direction = current_direction
                    self._contour_repeat_k = 1
                print(
                    f"Contour return: {type(action).__name__} "
                    f"idx={self._contour_idx}/{contour_points.shape[0]} "
                    f"nearest={nearest_distance:.2f} area={area:.2f} "
                    f"attempt={attempt} retries={retries} k={self._contour_repeat_k}"
                )
                return action
            if action is not None and self._is_opposite_action(action):
                print(
                    f"Contour skip opposite: {type(action).__name__} "
                    f"idx={self._contour_idx} attempt={attempt} retries={retries}"
                )
                self._contour_idx = (
                    self._contour_idx
                    + self.contour_direction * self.contour_lookahead * (attempt + 2)
                ) % contour_points.shape[0]
                continue
            self._contour_idx = (
                self._contour_idx + self.contour_direction * self.contour_lookahead
            ) % contour_points.shape[0]

        print(
            f"Contour fallback: no valid contour action after retries "
            f"(n={contour_points.shape[0]}, area={area:.2f})"
        )
        return super().dynamic_call(state)
