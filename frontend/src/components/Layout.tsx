import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { useI18n } from "../i18n/I18nContext";
import { useTheme } from "../context/ThemeContext";
import {
  DatabaseIcon,
  GlobeIcon,
  HistoryIcon,
  HomeIcon,
  HubIcon,
  LibraryIcon,
  MenuIcon,
  MoonIcon,
  SearchIcon,
  SettingsIcon,
  ShieldIcon,
  ShortsIcon,
  SubsIcon,
  SunIcon,
  WatchIcon,
  YourVideosIcon,
} from "./icons";

const DATABASE_URL =
  "https://supabase.com/dashboard/project/kxzxqlhdzkpcanotdbgl/editor/17547?schema=public&sort=created_at%3Adesc";

export function Layout() {
  const { t, toggleLocale } = useI18n();
  const { theme, toggleTheme } = useTheme();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);

  const navLinkClass = ({ isActive }: { isActive: boolean }) =>
    isActive ? "nav-item active" : "nav-item";

  const closeMobile = () => setSidebarOpen(false);

  const sidebarClass = [
    "sidebar",
    sidebarOpen ? "open" : "",
    collapsed ? "collapsed" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={collapsed ? "app-shell collapsed-shell" : "app-shell"}>
      <nav className={sidebarClass}>
        <NavLink to="/" className="brand" onClick={closeMobile} aria-label="SignalMod">
          <img src="/signalmod_logo.png" alt="SignalMod" className="brand-logo" />
        </NavLink>

        <div className="sidebar-section">
          <NavLink to="/" end className={navLinkClass} onClick={closeMobile} title={t.nav.home}>
            <span className="nav-icon"><HomeIcon /></span>
            <span className="nav-label">{t.nav.home}</span>
          </NavLink>
          <button className="nav-item" type="button" disabled title={t.nav.shorts}>
            <span className="nav-icon"><ShortsIcon /></span>
            <span className="nav-label">{t.nav.shorts}</span>
          </button>
          <button className="nav-item" type="button" disabled title={t.nav.subscriptions}>
            <span className="nav-icon"><SubsIcon /></span>
            <span className="nav-label">{t.nav.subscriptions}</span>
          </button>
        </div>

        <div className="sidebar-section">
          <button className="nav-item" type="button" title={t.nav.library}>
            <span className="nav-icon"><LibraryIcon /></span>
            <span className="nav-label">{t.nav.library}</span>
          </button>
          <button className="nav-item" type="button" title={t.nav.history}>
            <span className="nav-icon"><HistoryIcon /></span>
            <span className="nav-label">{t.nav.history}</span>
          </button>
          <button className="nav-item" type="button" title={t.nav.yourVideos}>
            <span className="nav-icon"><YourVideosIcon /></span>
            <span className="nav-label">{t.nav.yourVideos}</span>
          </button>
        </div>

        <div className="sidebar-section">
          <h3 className="sidebar-heading">
            <span className="nav-icon"><ShieldIcon /></span>
            <span className="nav-label">{t.nav.moderation}</span>
          </h3>
          <NavLink to="/" end className={navLinkClass} onClick={closeMobile} title={t.nav.watch}>
            <span className="nav-icon"><WatchIcon /></span>
            <span className="nav-label">{t.nav.watch}</span>
          </NavLink>
          <NavLink to="/hub" className={navLinkClass} onClick={closeMobile} title={t.nav.hub}>
            <span className="nav-icon"><HubIcon /></span>
            <span className="nav-label">{t.nav.hub}</span>
          </NavLink>
          <NavLink to="/settings" className={navLinkClass} onClick={closeMobile} title={t.nav.settings}>
            <span className="nav-icon"><SettingsIcon /></span>
            <span className="nav-label">{t.nav.settings}</span>
          </NavLink>
          <a
            className="nav-item"
            href={DATABASE_URL}
            target="_blank"
            rel="noopener noreferrer"
            onClick={closeMobile}
            title={t.nav.database}
          >
            <span className="nav-icon"><DatabaseIcon /></span>
            <span className="nav-label">{t.nav.database}</span>
          </a>
        </div>
      </nav>

      {sidebarOpen && (
        <div className="sidebar-backdrop" onClick={closeMobile} />
      )}

      <main className="main-content">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="icon-btn"
              type="button"
              onClick={() => {
                if (window.matchMedia("(max-width: 820px)").matches) {
                  setSidebarOpen((v) => !v);
                } else {
                  setCollapsed((v) => !v);
                }
              }}
              aria-label={t.topbar.toggleSidebar}
              title={t.topbar.toggleSidebar}
            >
              <MenuIcon />
            </button>
          </div>

          <div className="topbar-search">
            <SearchIcon />
            <input
              placeholder={t.topbar.searchPlaceholder}
              aria-label={t.topbar.searchPlaceholder}
            />
          </div>

          <div className="topbar-right">
            <button
              className="lang-toggle"
              type="button"
              onClick={toggleLocale}
              title={t.topbar.toggleLanguage}
              aria-label={t.topbar.toggleLanguage}
            >
              <GlobeIcon /> {t.topbar.languageLabel}
            </button>
            <button
              className="theme-toggle"
              type="button"
              onClick={toggleTheme}
              title={t.topbar.toggleTheme}
              aria-label={t.topbar.toggleTheme}
            >
              {theme === "dark" ? <SunIcon /> : <MoonIcon />}
            </button>
            <button className="sign-in" type="button">
              {t.topbar.signIn}
            </button>
          </div>
        </header>

        <Outlet />
      </main>
    </div>
  );
}
