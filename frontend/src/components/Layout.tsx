import { NavLink, Outlet } from "react-router-dom";
import { ModelBanner } from "./ModelBanner";

export function Layout() {
  return (
    <div className="app-shell">
      <nav className="sidebar">
        <div className="logo">youtube_hate_detector</div>
        <NavLink to="/" end className={({ isActive }) => (isActive ? "nav active" : "nav")}>
          Watch
        </NavLink>
        <NavLink to="/hub" className={({ isActive }) => (isActive ? "nav active" : "nav")}>
          Moderator Hub
        </NavLink>
        <NavLink to="/settings" className={({ isActive }) => (isActive ? "nav active" : "nav")}>
          Settings
        </NavLink>
      </nav>
      <main className="main-content">
        {/* <ModelBanner /> */}
        <Outlet />
      </main>
    </div>
  );
}
