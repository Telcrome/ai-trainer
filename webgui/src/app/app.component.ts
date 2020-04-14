import { Component } from '@angular/core';
import { LogServiceService } from './log-service.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  constructor(private logs: LogServiceService) { }

  title = this.logs.welcome;
  logList = this.logs.getLogs();
  model = {
    left: true,
    middle: false,
    right: false
  };
  
}
