"""
Session management service for privacy budget tracking.
"""

import secrets
from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models import PrivacySession
from ..config import get_settings

settings = get_settings()


class SessionService:
    """Manages privacy sessions and budget tracking."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_session(self, initial_budget: Optional[float] = None) -> PrivacySession:
        """
        Create a new privacy session with the specified budget.
        
        Args:
            initial_budget: Starting privacy budget (default from settings)
            
        Returns:
            The created PrivacySession
        """
        budget = initial_budget or settings.default_session_budget
        
        session = PrivacySession(
            session_token=secrets.token_urlsafe(32),
            initial_budget=budget,
            remaining_budget=budget,
            created_at=datetime.utcnow(),
            is_active=True,
            query_history=[],
        )
        
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        
        return session
    
    async def get_session(self, token: str) -> Optional[PrivacySession]:
        """Get a session by its token."""
        result = await self.db.execute(
            select(PrivacySession).where(PrivacySession.session_token == token)
        )
        return result.scalar_one_or_none()
    
    async def spend_budget(
        self,
        token: str,
        epsilon: float,
        query_info: Optional[dict] = None
    ) -> PrivacySession:
        """
        Deduct privacy budget from a session.
        
        Args:
            token: Session token
            epsilon: Amount of budget to spend
            query_info: Optional metadata about the query
            
        Returns:
            Updated session
            
        Raises:
            ValueError: If session not found or insufficient budget
        """
        session = await self.get_session(token)
        
        if not session:
            raise ValueError("Session not found")
        
        if not session.is_active:
            raise ValueError("Session is no longer active")
        
        if session.remaining_budget < epsilon:
            raise ValueError(
                f"Insufficient budget. Remaining: {session.remaining_budget}, "
                f"Required: {epsilon}"
            )
        
        # Update session
        session.remaining_budget -= epsilon
        session.last_query_at = datetime.utcnow()
        
        # Record query in history
        query_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "epsilon": epsilon,
        }
        if query_info:
            query_record.update(query_info)
        
        if session.query_history is None:
            session.query_history = []
        session.query_history = session.query_history + [query_record]
        
        await self.db.commit()
        await self.db.refresh(session)
        
        return session
    
    async def deactivate_session(self, token: str) -> bool:
        """Deactivate a session (no more queries allowed)."""
        session = await self.get_session(token)
        
        if not session:
            return False
        
        session.is_active = False
        await self.db.commit()
        
        return True
    
    async def get_session_status(self, token: str) -> Optional[dict]:
        """Get current session status."""
        session = await self.get_session(token)
        
        if not session:
            return None
        
        return {
            "remaining_budget": session.remaining_budget,
            "queries_made": len(session.query_history) if session.query_history else 0,
            "is_active": session.is_active,
            "can_query": session.is_active and session.remaining_budget > 0,
        }

