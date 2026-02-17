from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
)

from app.model.entities import Barrier, GridSettings, Source


class Canvas2D(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.sources: list[Source] = []
        self.barriers: list[Barrier] = []
        self.grid: GridSettings | None = None
        self.mode = "select"
        self._new_barrier_pts: list[tuple[float, float]] = []
        self.setRenderHints(self.renderHints())

    def set_data(self, sources: list[Source], barriers: list[Barrier], grid: GridSettings | None = None) -> None:
        self.sources = sources
        self.barriers = barriers
        self.grid = grid
        self.redraw()

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def redraw(self, contours=None) -> None:
        self.scene.clear()
        pen_grid = QPen(Qt.lightGray)
        if self.grid and self.grid.nx > 0 and self.grid.ny > 0:
            for i in range(self.grid.nx + 1):
                x = self.grid.xmin + i * self.grid.cell_size
                self.scene.addLine(x, self.grid.ymin, x, self.grid.ymax, pen_grid)
            for j in range(self.grid.ny + 1):
                y = self.grid.ymin + j * self.grid.cell_size
                self.scene.addLine(self.grid.xmin, y, self.grid.xmax, y, pen_grid)

        for s in self.sources:
            p = QGraphicsEllipseItem(s.x - 1.5, s.y - 1.5, 3.0, 3.0)
            p.setBrush(QBrush(Qt.red))
            p.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
            p.setData(0, ("source", s.source_id))
            self.scene.addItem(p)
            txt = QGraphicsTextItem(str(s.source_id))
            txt.setPos(QPointF(s.x + 2, s.y + 2))
            self.scene.addItem(txt)

        for b in self.barriers:
            for i in range(len(b.points) - 1):
                x1, y1 = b.points[i]
                x2, y2 = b.points[i + 1]
                li = QGraphicsLineItem(x1, y1, x2, y2)
                li.setPen(QPen(Qt.black, 1.5))
                self.scene.addItem(li)

        if contours:
            for lvl, seg in contours:
                if len(seg) < 2:
                    continue
                for i in range(len(seg) - 1):
                    self.scene.addLine(seg[i][0], seg[i][1], seg[i + 1][0], seg[i + 1][1], QPen(Qt.blue, 0.8))

        self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        pos = self.mapToScene(event.position().toPoint())
        if self.mode == "add_source" and event.button() == Qt.LeftButton:
            new_id = len(self.sources) + 1
            self.sources.append(Source(source_id=new_id, x=pos.x(), y=pos.y(), z=1.0, lwa_total=90.0))
            self.redraw()
        elif self.mode == "add_barrier" and event.button() == Qt.LeftButton:
            self._new_barrier_pts.append((pos.x(), pos.y()))
            if len(self._new_barrier_pts) > 1:
                p1 = self._new_barrier_pts[-2]
                p2 = self._new_barrier_pts[-1]
                self.scene.addLine(p1[0], p1[1], p2[0], p2[1], QPen(Qt.darkGray, 2))
        elif self.mode == "add_barrier" and event.button() == Qt.RightButton:
            if len(self._new_barrier_pts) > 1:
                bid = len(self.barriers) + 1
                self.barriers.append(Barrier(barrier_id=bid, points=self._new_barrier_pts.copy(), height=2.0))
            self._new_barrier_pts = []
            self.redraw()
        else:
            super().mousePressEvent(event)
