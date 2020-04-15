import { Component, OnInit } from '@angular/core';
import { ConfigService } from './config.service';
import { SocketioService } from './socketio.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  constructor(private config: ConfigService, private socketService: SocketioService) { }

  ngOnInit(): void {
    if (this.config.serverCreds == null) {
      // Ask the user for credentials of the server to connect to
      this.config.serverCreds = { url: 'http://localhost:5000' };
      this.socketService.setupSocketConnection();
    }
  }

}
